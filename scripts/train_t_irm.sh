#!/usr/bin/env bash
python main_irm.py \
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
--use_model DELU_DDG \
--max_iter 5000 \
--dataset SampleDataset \
--weight_decay 0.001 \
--action_threshold 0.5 \
--background_threshold 0.5 \
--top_k_rat 10 \
--similarity_threshold 0.8 \
--AWM DDG_Net \
--model_name Thumos14-DDG_Net \
--alpha6 1 \
--temperature 0.5 \
--weight 2 \
--alpha5 3.2

: << results
#!/usr/bin/env bash
python main_irm.py \
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
--AWM BWA_fusion_dropout_feat_v2 \
--work_dir work_dir/t-ddg-irm \