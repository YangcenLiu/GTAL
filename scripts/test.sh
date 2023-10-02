#!/usr/bin/env bash
python test.py \
--seed 3 \
--use_model DELU_MULTI_SCALE \
--model_name delu_multi_scale \
--proposal_method multi_scale_multiple_threshold_hamnet \
--action_threshold 0.5 \
--background_threshold 0.5 \
--similarity_threshold 0.8 \
--top_k_rat 10 \
--scale 1 \


