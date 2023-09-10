python main.py \
--model_name DELU_MULTI_SCALE \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--rat_atn 5 \
--k 5 \
--eval_interval 20 \
--dataset_name ActivityNet1.2 \
--path_dataset /data0/lixunsong/Datasets/ActivityNet1.2 \
--num_class 100 \
--use_model DELU_MULTI_SCALE \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 30000 \
--class_mapping a2t_class_mapping.json \
--proposal_method multi_scale_multiple_threshold_hamnet \
--scales 1 3 7 15 31 \
--work_dir work_dir/multi_scale_anet_all_scale

'''
lyc4 1 3 7 15 31

'''