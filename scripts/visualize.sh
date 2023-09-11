#!/usr/bin/env bash
python visualize.py \
--model_name DELU \
--seed 0 \
--max_seqlen 320 \
--dataset_name Thumos14reduced \
--path_dataset /data0/lixunsong/Datasets/THUMOS14 \
--dataset SampleDataset \