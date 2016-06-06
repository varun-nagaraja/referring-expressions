#!/usr/bin/env bash

GPU_ID=0

WEIGHTS=./caffe/models/vggnet/VGG_ILSVRC_16_layers.caffemodel

EXP_NAME=baseline
#EXP_NAME=max_margin
#EXP_NAME=mil_context_withNegMargin
#EXP_NAME=mil_context_withPosNegMargin

SOLVER_PROTO=./proto_files/unc_refexp/unc_refexp.$EXP_NAME.solver.prototxt

./build/tools/caffe train \
    -solver $SOLVER_PROTO \
    -gpu $GPU_ID \
    -weights $WEIGHTS \
    |& tee proto_files/unc_refexp/unc_refexp.$EXP_NAME.log
