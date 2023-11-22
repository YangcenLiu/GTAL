python main.py \
--model_name DELU_MULTI_SCALE \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--rat_atn 5 \
--k 5 \
--eval_interval 200 \
--dataset_name ActivityNet1.2 \
--path_dataset /data0/lixunsong/Datasets/ActivityNet1.2 \
--num_class 100 \
--use_model DELU_MULTI_SCALE \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 22000000000000000000000000000000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--proposal_method multi_scale_multiple_threshold_hamnet \
--work_dir work_dir/gift4liupeng

'''
python main.py \
--model_name DELU_DDG_MULTI_SCALE \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--rat_atn 5 \
--k 5 \
--eval_interval 200 \
--dataset_name ActivityNet1.2 \
--path_dataset /data0/lixunsong/Datasets/ActivityNet1.2 \
--num_class 100 \
--use_model DELU_DDG_MULTI_SCALE \
--dataset AntSampleDataset \
--AWM DDG_Net \
--action_threshold 0.5 \
--background_threshold 0.5 \
--top_k_rat 10 \
--similarity_threshold 0.8 \
--alpha6 0.5 \
--temperature 0.5 \
--weight 2 \
--alpha5 1.7 \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 22000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--proposal_method multi_scale_multiple_threshold_hamnet \
--work_dir work_dir/anet_ddg_multi

lyc4 1 3 7 15 31

Number of ground truth instances: 3358
Number of predictions: 10881
Classification map 88.712655
map @ 0.1 = 48.381 ||map @ 0.2 = 35.334 ||map @ 0.3 = 22.797 ||map @ 0.4 = 15.860 ||map @ 0.5 = 10.875 ||map @ 0.6 = 6.322 ||map @ 0.7 = 3.218 ||map @ 0.8 = 1.239 ||map @ 0.9 = 0.179 
mAP Avg ALL: 16.023
MAX map @ 0.1 = 49.435 ||MAX map @ 0.2 = 35.700 ||MAX map @ 0.3 = 23.446 ||MAX map @ 0.4 = 16.262 ||MAX map @ 0.5 = 11.212 ||MAX map @ 0.6 = 6.476 ||MAX map @ 0.7 = 3.297 ||MAX map @ 0.8 = 1.116 ||MAX map @ 0.9 = 0.167 
mAP Avg 0.1-0.5: 27.21094340880778, mAP Avg 0.1-0.7: 20.832453316620402, mAP Avg ALL: 16.345624046976344

OOD classification map 91.069404
map @ 0.5 = 25.494 ||map @ 0.55 = 21.308 ||map @ 0.6 = 16.200 ||map @ 0.65 = 12.623 ||map @ 0.7 = 10.825 ||map @ 0.75 = 7.331 ||map @ 0.8 = 5.809 ||map @ 0.85 = 3.182 ||map @ 0.9 = 1.603 ||map @ 0.95 = 0.395 
mAP Avg ALL: 10.477
MAX map @ 0.5 = 24.941 ||MAX map @ 0.55 = 21.158 ||MAX map @ 0.6 = 16.885 ||MAX map @ 0.65 = 13.379 ||MAX map @ 0.7 = 11.041 ||MAX map @ 0.75 = 7.473 ||MAX map @ 0.8 = 5.741 ||MAX map @ 0.85 = 3.962 ||MAX map @ 0.9 = 2.497 ||MAX map @ 0.95 = 0.423 
mAP Avg 0.5-0.95: 10.749972444757864
'''