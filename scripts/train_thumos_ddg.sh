#!/usr/bin/env bash
python main.py \
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
MAX map @ 0.1 = 72.262 ||MAX map @ 0.2 = 68.159 ||MAX map @ 0.3 = 58.044 ||MAX map @ 0.4 = 49.040 ||MAX map @ 0.5 = 40.934 ||MAX map @ 0.6 = 26.600 ||MAX map @ 0.7 = 14.030 ||MAX map @ 0.8 = 7.409 ||MAX map @ 0.9 = 2.123 
mAP Avg 0.1-0.5: 57.687814589372344, mAP Avg 0.1-0.7: 47.00995501841288, mAP Avg ALL: 37.622430819701485



class DirtyAvgPool(nn.Module):
        def __init__(self, scale):
            super(DirtyAvgPool, self).__init__()
            self.scale = scale
            self.padding = nn.ZeroPad2d((self.scale//2, self.scale//2, 0, 0))

        def forward(self, x):
            if self.scale == 1:
                return x

            b, e, t = x.size()
            y = x
            # print(y.size())
            x = self.padding(x)
            # print(x.size())
            for i in range(self.scale//2, x.size(-1)-self.scale//2):
                start = i - self.scale//2
                end = i + self.scale//2+1
                window = x[:,:,start:end]

                # print(window.size(), self.scale)
                center=x[:,:,i]
                left = window[:,:,:self.scale//2].max(dim=-1)[0]
                right = window[:,:,-self.scale//2:].max(dim=-1)[0]
                replace = torch.min(left, right)
                mask = center < replace
                # center[mask] = center[mask] + (replace[mask] - center[mask]) * 0.8
                center[mask] = center[mask] - (replace[mask] - center[mask]) * 1.0
                y[:,:,i-self.scale//2] = center
            return y
