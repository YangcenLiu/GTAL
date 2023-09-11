#!/usr/bin/env bash
python main.py \
--model_name DELU_MULTI_SCALE \
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
--use_model DELU_MULTI_SCALE \
--max_iter 5000 \
--dataset SampleDataset \
--weight_decay 0.001 \
--AWM BWA_fusion_dropout_feat_v2 \
--proposal_method multi_scale_multiple_threshold_hamnet \
--scales 13 \
--work_dir work_dir/thumos_13

'''
scales 13



scales 1 3
MAX map @ 0.1 = 69.763 ||MAX map @ 0.2 = 63.990 ||MAX map @ 0.3 = 55.176 ||MAX map @ 0.4 = 46.474 ||MAX map @ 0.5 = 37.959 ||MAX map @ 0.6 = 26.264 ||MAX map @ 0.7 = 14.312 ||MAX map @ 0.8 = 7.371 ||MAX map @ 0.9 = 2.135 
mAP Avg 0.1-0.5: 54.67237976387149, mAP Avg 0.1-0.7: 44.84823094933891, mAP Avg ALL: 35.93821882222915

MAX map @ 0.5 = 10.445 ||MAX map @ 0.55 = 7.961 ||MAX map @ 0.6 = 6.784 ||MAX map @ 0.65 = 5.044 ||MAX map @ 0.7 = 4.265 ||MAX map @ 0.75 = 3.278 ||MAX map @ 0.8 = 2.007 ||MAX map @ 0.85 = 0.895 ||MAX map @ 0.9 = 0.235 ||MAX map @ 0.95 = 0.082 
mAP Avg 0.5-0.95: 4.099572145102045

scales 1 3 7

MAX map @ 0.1 = 69.277 ||MAX map @ 0.2 = 62.785 ||MAX map @ 0.3 = 54.477 ||MAX map @ 0.4 = 45.278 ||MAX map @ 0.5 = 37.114 ||MAX map @ 0.6 = 24.350 ||MAX map @ 0.7 = 13.815 ||MAX map @ 0.8 = 6.613 ||MAX map @ 0.9 = 1.954 
mAP Avg 0.1-0.5: 53.78627033839691, mAP Avg 0.1-0.7: 43.870842216648924, mAP Avg ALL: 35.073583451615015

MAX map @ 0.5 = 13.339 ||MAX map @ 0.55 = 10.156 ||MAX map @ 0.6 = 8.563 ||MAX map @ 0.65 = 6.777 ||MAX map @ 0.7 = 5.249 ||MAX map @ 0.75 = 4.334 ||MAX map @ 0.8 = 2.773 ||MAX map @ 0.85 = 1.336 ||MAX map @ 0.9 = 0.361 ||MAX map @ 0.95 = 0.151 
mAP Avg 0.5-0.95: 5.303877325985442


scales 1 3 7 15 31
MAX map @ 0.1 = 67.239 ||MAX map @ 0.2 = 59.774 ||MAX map @ 0.3 = 51.027 ||MAX map @ 0.4 = 42.671 ||MAX map @ 0.5 = 35.237 ||MAX map @ 0.6 = 23.196 ||MAX map @ 0.7 = 12.797 ||MAX map @ 0.8 = 5.952 ||MAX map @ 0.9 = 1.622 
mAP Avg 0.1-0.5: 51.189703819974696, mAP Avg 0.1-0.7: 41.70595659838838, mAP Avg ALL: 33.279464947074125
mAP Avg 0.1-0.5: 51.189703819974696, mAP Avg 0.1-0.7: 41.70595659838838, mAP Avg ALL: 33.279464947074125


MAX map @ 0.5 = 16.235 ||MAX map @ 0.55 = 12.244 ||MAX map @ 0.6 = 10.580 ||MAX map @ 0.65 = 8.348 ||MAX map @ 0.7 = 6.287 ||MAX map @ 0.75 = 5.113 ||MAX map @ 0.8 = 3.559 ||MAX map @ 0.85 = 1.501 ||MAX map @ 0.9 = 0.588 ||MAX map @ 0.95 = 0.203 
mAP Avg 0.5-0.95: 6.465655580637207
'''