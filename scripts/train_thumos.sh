#!/usr/bin/env bash
python main.py \
--model_name DELU \
--seed 0 \
--alpha_edl 1.3 \
--alpha_uct_guide 0.4 \
--amplitude 0.7 \
--alpha2 0.4 \
--eval_interval 50 \
--max_seqlen 320 \
--lr 0.00005 \
--k 7 \
--dataset_name Thumos14reduced \
--path_dataset /data0/lixunsong/Datasets/THUMOS14 \
--num_class 20 \
--use_model DELU \
--max_iter 5000 \
--dataset SampleDataset \
--weight_decay 0.001 \
--AWM BWA_fusion_dropout_feat_v2

'''
Number of ground truth instances: 3358
Number of predictions: 36301
Classification map 94.848915
MAX map @ 0.1 = 70.877 ||MAX map @ 0.2 = 65.703 ||MAX map @ 0.3 = 56.182 ||MAX map @ 0.4 = 47.231 ||MAX map @ 0.5 = 39.923 ||MAX map @ 0.6 = 27.158 ||MAX map @ 0.7 = 14.481 ||MAX map @ 0.8 = 7.369 ||MAX map @ 0.9 = 1.949 
mAP Avg 0.1-0.5: 55.98292387871897, mAP Avg 0.1-0.7: 45.93623030604365, mAP Avg ALL: 36.763526042316215

Number of ground truth instances: 371
Number of predictions: 10825
OOD classification map 96.071318
map @ 0.5 = 10.374 ||map @ 0.55 = 7.401 ||map @ 0.6 = 6.469 ||map @ 0.65 = 4.618 ||map @ 0.7 = 3.451 ||map @ 0.75 = 2.518 ||map @ 0.8 = 1.660 ||map @ 0.85 = 0.793 ||map @ 0.9 = 0.425 ||map @ 0.95 = 0.170 
mAP Avg ALL: 3.788
'''