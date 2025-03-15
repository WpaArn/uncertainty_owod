#!/bin/bash

BENCHMARK=${BENCHMARK:-"S-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}
EXP=${EXP:-"exp4"}
if [ $BENCHMARK == "M-OWODB" ]; then
   python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0019999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0034999.pth

 python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0049999.pth

 python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0064999.pth

 python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0079999.pth

 python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0094999.pth
else
#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml

#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0039999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0064999.pth

#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0069999.pth

#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0084999.pth
#
#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0099999.pth
#
#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --resume MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_00114999.pth
fi