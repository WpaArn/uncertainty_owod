import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.NUM_CLASSES
        d_model = cfg.MODEL.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        nhead = cfg.MODEL.NHEADS
        dropout = cfg.MODEL.DROPOUT
        activation = cfg.MODEL.ACTIVATION
        num_heads = cfg.MODEL.NUM_HEADS
        self.head_series = nn.ModuleList([
            RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
            for i in range(num_heads)
        ])
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.DEEP_SUPERVISION

        # Gaussian random feature embedding layer for time
        self.d_model = d_model

        # Init parameters.
        self.num_classes = num_classes
        prior_prob = cfg.MODEL.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, init_features):

        inter_class_logits = []
        inter_objectness = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, objectness, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features,
                                                                           self.box_pooler)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_objectness.append(objectness)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_objectness), torch.stack(inter_pred_bboxes)

        return class_logits[None], objectness[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model
        self.disentangled = cfg.MODEL.DISENTANGLED

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            if self.disentangled != 2:  # avoid partial sphere
                cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.class_logits = nn.Linear(d_model, num_classes)
        if self.disentangled == 1:  # separate head
            self.object_logit = nn.Linear(d_model, 1)
        elif self.disentangled == 2:  # feature orthogonality
            self.norm4 = nn.LayerNorm(d_model)
            self.object_logit = nn.BatchNorm1d(1)

            # initialize class splits and calibration layer
            self.classes = list(cfg.TEST.PREV_CLASSES)
            if len(self.classes) > 0:
                self.calibration = [nn.Linear(d_model, d_model) for _ in self.classes]
                for ly in self.calibration:
                    nn.init.eye_(ly.weight)
                    nn.init.zeros_(ly.bias)
                self.calibration = nn.ModuleList(self.calibration + [nn.Identity()])
            self.classes += [cfg.MODEL.NUM_CLASSES - sum(self.classes)]
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes,
                                                                                             self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        if self.disentangled == 0:
            class_logits = self.class_logits(cls_feature)
            objectness = torch.sigmoid(class_logits).sum(dim=-1)  # actually not used
        elif self.disentangled == 1:  # separate head
            class_logits = self.class_logits(cls_feature)
            objectness = torch.sigmoid(self.object_logit(cls_feature))
        else:  # feature orthogonality
            # apply calibration layer to maintain orthogonality
            if len(self.classes) > 1:
                cls_features = [self.calibration[i](cls_feature) for i in range(len(self.classes))]
                cls_feature = torch.stack(cls_features, dim=1)
                cls_feature = rearrange(cls_feature, 'b n c -> (b n) c')

            obj_feature = cls_feature.norm(dim=-1, keepdim=True)
            cls_feature = self.norm4(cls_feature)
            class_logits = self.class_logits(cls_feature)
            class_logit_max = class_logits[:, :-1].max(dim=-1, keepdim=True).values
            class_logits -= class_logit_max
            tmp = class_logits[:, :-1].exp().sum(dim=-1, keepdim=True)
            # assign unknown logit s.t. p(unknown) = p_original(known - max_known)
            unknown_logit = (tmp - 1).log() + tmp.log() - (1 + class_logits[:, -1:].exp()).log()
            class_logits = torch.cat([class_logits[:, :-1], unknown_logit], dim=-1)
            objectness = torch.sigmoid(self.object_logit(obj_feature))

            # routing within the calibration layer
            if len(self.classes) > 1:
                class_logits = rearrange(class_logits, '(b n) c -> b n c', n=len(self.classes))
                objectness = rearrange(objectness, '(b n) 1 -> b n 1', n=len(self.classes))
                class_logits_splits = torch.split(class_logits, self.classes, dim=-1)
                weight_cat = []
                for i in range(len(self.classes)):
                    weight_cat.append(class_logits_splits[i][:, i, :].max(dim=-1, keepdim=True)[0])
                routing_weights = F.gumbel_softmax(torch.cat(weight_cat, dim=-1), dim=-1, hard=True)
                routing_weights = routing_weights.unsqueeze(dim=-1)
                class_logits = (routing_weights * class_logits).sum(dim=1)
                objectness = (routing_weights * objectness).sum(dim=1)

        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(N, nr_boxes, -1), objectness.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")