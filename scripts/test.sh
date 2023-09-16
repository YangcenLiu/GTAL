#!/usr/bin/env bash
python test.py \
--use_model DELU_MULTI_SCALE \
--model_name delu_multi_scale \
--proposal_method multi_scale_multiple_threshold_hamnet \


'''

使用公有的8类结果

Thumos -> 1.3
IND
MAX map @ 0.1 = 81.053 ||MAX map @ 0.2 = 77.915 ||MAX map @ 0.3 = 71.116 ||MAX map @ 0.4 = 64.163 ||MAX map @ 0.5 = 53.615 ||MAX map @ 0.6 = 37.563 ||MAX map @ 0.7 = 22.079 ||MAX map @ 0.8 = 11.561 ||MAX map @ 0.9 = 2.234 
Thumos14: mAP Avg 0.1-0.5: 69.5724527383941, mAP Avg 0.1-0.7: 58.21486302943922, mAP Avg ALL: 46.810974855631414

OOD classification map 93.025982
map @ 0.5 = 27.367 ||map @ 0.55 = 24.507 ||map @ 0.6 = 22.039 ||map @ 0.65 = 16.630 ||map @ 0.7 = 13.500 ||map @ 0.75 = 9.142 ||map @ 0.8 = 6.808 ||map 
@ 0.85 = 4.341 ||map @ 0.9 = 2.358 ||map @ 0.95 = 0.546 
mAP Avg ALL: 12.724

1.2 -> 1.3
IND
Classification map 91.209600                                                                                                                            
map @ 0.5 = 31.649 ||map @ 0.55 = 27.310 ||map @ 0.6 = 25.500 ||map @ 0.65 = 20.520 ||map @ 0.7 = 17.667 ||map @ 0.75 = 15.315 ||map @ 0.8 = 10.849 ||ma
p @ 0.85 = 7.693 ||map @ 0.9 = 4.149 ||map @ 0.95 = 1.165                                                                                               
mAP Avg ALL: 16.182                                                                                                                                                                                           
ActivityNet1.2: mAP Avg 0.5-0.95: 16.181662430481936                                                                                               
                    
OOD classification map 97.063984
map @ 0.5 = 36.205 ||map @ 0.55 = 32.571 ||map @ 0.6 = 29.580 ||map @ 0.65 = 24.436 ||map @ 0.7 = 21.130 ||map @ 0.75 = 17.627 ||map @ 0.8 = 12.217 
||map @ 0.85 = 8.182 ||map @ 0.9 = 4.689 ||map @ 0.95 = 1.849 
mAP Avg ALL: 18.849
MAX map @ 0.5 = 36.205 ||MAX map @ 0.55 = 32.571 ||MAX map @ 0.6 = 29.580 ||MAX map @ 0.65 = 24.436 ||MAX map @ 0.7 = 21.130 ||MAX map @ 0.75 = 17.627 ||MAX map @ 0.8 = 12.217 ||MAX map @ 0.85 = 8.182 ||MAX map @ 0.9 = 4.689 ||MAX map @ 0.95 = 1.849 
ActivityNet1.3: mAP Avg 0.5-0.95: 18.84868590995461
'''