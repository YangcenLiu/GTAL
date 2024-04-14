python main_adapter.py \
--model_name DELU_Adapter \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--eval_interval 5 \
--rat_atn 5 \
--k 5 \
--lr 0.00003 \
--batch_size 30 \
--dataset_name ActivityNet1.2 \
--path_dataset /data0/lixunsong/Datasets/ActivityNet1.2 \
--num_class 100 \
--use_model DELU_Adapter \
--dataset AntSampleDataset \
--weight_decay 0.0005 \
--max_seqlen 60 \
--max_iter 1000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--pretrained_ckpt ckpt/best_delu_act.pkl \
--AWM BWA_fusion_dropout_feat_v2 \
--refine_scale 7 \
--refine_alpha 1.4 \
--work_dir work_dir/a2t-adapter \




: << results
python main_adapter.py \
--model_name DELU_Adapter \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--eval_interval 1 \
--rat_atn 5 \
--k 5 \
--lr 0.00003 \
--batch_size 30 \
--dataset_name ActivityNet1.2 \
--path_dataset /data0/lixunsong/Datasets/ActivityNet1.2 \
--num_class 100 \
--use_model DELU_Adapter \
--dataset AntSampleDataset \
--weight_decay 0.0005 \
--max_seqlen 60 \
--max_iter 1000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--pretrained_ckpt ckpt/best_ddg_act.pkl \
--AWM DDG_Net \
--refine_scale 7 \
--refine_alpha 1.4 \
--work_dir work_dir/a-ind-ddg-adapter \




python main_adapter.py \
--model_name DELU_Adapter \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--eval_interval 5 \
--rat_atn 5 \
--k 5 \
--lr 0.00003 \
--batch_size 30 \
--dataset_name ActivityNet1.2 \
--path_dataset /data0/lixunsong/Datasets/ActivityNet1.2 \
--num_class 100 \
--use_model DELU_Adapter \
--dataset AntSampleDataset \
--weight_decay 0.0005 \
--max_seqlen 60 \
--max_iter 1000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--pretrained_ckpt ckpt/best_delu_act.pkl \
--AWM BWA_fusion_dropout_feat_v2 \
--refine_scale 7 \
--refine_alpha 1.5 \
--work_dir work_dir/a-ind-adapter \



lr 1e-5
update 0.9
7 1.5
MAX map @ 0.1 = 73.474 ||MAX map @ 0.2 = 71.090 ||MAX map @ 0.3 = 66.703 ||MAX map @ 0.4 = 59.838 ||MAX map @ 0.5 = 49.565 ||MAX map @ 0.6 = 34.600 ||MAX map @ 0.7 
= 19.853 ||MAX map @ 0.8 = 9.155 ||MAX map @ 0.9 = 1.545                                                                                                            
mAP Avg 0.1-0.7: 53.589116971561246 