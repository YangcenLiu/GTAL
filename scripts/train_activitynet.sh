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
--use_model DELU_ACT \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 22000 \
--class_mapping class_mapping/a2t_class_mapping.json \
--work_dir work_dir/anet_seed1

'''
Number of ground truth instances: 3582                                                                                                                            
Number of predictions: 42496                                                                                                                                      
Classification map 91.249684 
MAX map @ 0.5 = 43.814 ||MAX map @ 0.55 = 39.870 ||MAX map @ 0.6 = 36.487 ||MAX map @ 0.65 = 33.075 ||MAX map @ 0.7 = 29.448 ||MAX map @ 0.75 = 26.348 ||MAX map @ 0.8 = 2
2.250 ||MAX map @ 0.85 = 17.102 ||MAX map @ 0.9 = 11.834 ||MAX map @ 0.95 = 5.255

Number of ground truth instances: 3358                                                                                                                            
Number of predictions: 3904 
OOD classification map 97.231499                                                                          
mAP Avg 0.1-0.5: 36.5387915228076, mAP Avg 0.1-0.7: 33.041746817751516, mAP Avg ALL: 26.548320653006837
MAX map @ 0.1 = 16.848 ||MAX map @ 0.2 = 12.531 ||MAX map @ 0.3 = 8.942 ||MAX map @ 0.4 = 5.810 ||MAX map @ 0.5 = 3.746 ||MAX map @ 0.6 = 1.925 ||MAX map @ 0.7 = 0.671 ||MAX map @ 0.8 = 0.212 ||MAX map @ 0.9 = 0.016 
mAP Avg 0.1-0.7: 7.210496617058693
'''
