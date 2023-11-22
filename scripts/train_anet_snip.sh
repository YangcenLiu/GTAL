python main.py \
--model_name DELU_ACT \
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
--use_model BASE \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 320 \
--max_iter 22000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--work_dir work_dir/anet_delu_snip

python main.py \
--model_name DELU_ACT \
--seed 1 \
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
--use_model DELU_DDG \
--AWM DDG_Net \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 22000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--work_dir work_dir/anet_ddg_snip

python main.py \
--model_name DELU_ACT \
--seed 1 \
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
--use_model DELU \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 22000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--work_dir work_dir/anet_delu_snip