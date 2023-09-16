from __future__ import print_function

import os
import random
import time

import numpy as np
import torch

from tqdm import tqdm

import models
import options
import wsad_dataset
from test import test
from train import train, ddp_train
from ood_test import ood_test

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):

    setup(rank, world_size)
    
    seed = args.seed
    print(f'=============seed: {seed}, pid: {os.getpid()}=============')
    torch.manual_seed(seed)
    
    device = torch.device("cuda", rank)
    
    dataset = getattr(wsad_dataset, args.dataset)(args)
    ood_dataset = None
    
    if 'Thumos' in args.dataset_name:
        max_map = [0] * 9
        ood_max_map = [0] * 10
    else:
        max_map = [0] * 10
    
    log_model_path = os.path.join(args.work_dir, 'logs', args.model_name)
    ckpt_path = os.path.join(args.work_dir, 'ckpt')
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, device=device, opt=args).to(device)
    
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_loss = 0
    lrs = [args.lr, args.lr / 5, args.lr / 5 / 5]

    model = DDP(model, device_ids=[rank])
    
    for itr in tqdm(range(args.max_iter), ncols=80):
        loss = ddp_train(itr, dataset, args, model, optimizer, device)
        total_loss += loss
        if itr % args.eval_interval == 0 and not itr == 0 and rank == 0:
            print("")
            print('Iteration: %d, Loss: %.5f' % (itr, total_loss / args.eval_interval))
            total_loss = 0
            if rank == 0:
            torch.save(model.module.state_dict(), ckpt_path + '/last_' + args.model_name + '.pkl')
            iou, dmap = test(itr, dataset, args, model, device)
            if 'Thumos' in args.dataset_name:
                cond = sum(dmap[:7]) > sum(max_map[:7])
            else:
                cond = np.mean(dmap) > np.mean(max_map)
            if cond:
                torch.save(model.module.state_dict(), ckpt_path + '/best_' + args.model_name + '.pkl')
                max_map = dmap
            
            print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i], max_map[i] * 100) for i in range(len(iou))]))
            max_map = np.array(max_map)
            print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5]) * 100,
                                                                                     np.mean(max_map[:7]) * 100,
                                                                                     np.mean(max_map) * 100))
            
            if "thumos" in args.dataset_name.lower():
                args.dataset_name = "ActivityNet1.2"
                args.dataset = "AntSampleDataset"
                args.num_class = 100
                args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
                args.max_seqlen = 60
                
                if ood_dataset is None:
                    ood_dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)
                
                ood_iou, ood_dmap, ood_mAP_Avg_ALL = ood_test(
                    ood_dataset,
                    args,
                    model,
                    device,
                    class_mapping=args.class_mapping,
                    save_activation=False,
                    itr=itr,
                )
                
                args.dataset_name = "Thumos14reduced"
                args.dataset = ("SampleDataset",)
                args.num_class = 20
                args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
                args.max_seqlen = 320
                
                if cond:
                    ood_max_map = ood_dmap
                
                print(
                "||".join(
                    [
                        "MAX map @ {} = {:.3f} ".format(ood_iou[i], ood_max_map[i] * 100)
                        for i in range(len(ood_iou))
                    ]
                    )
                )
                ood_max_map = np.array(ood_max_map)
                print('mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))
            
            if "activitynet" in args.dataset_name.lower():  # only support thumos14 for now

                args.dataset_name = "Thumos14reduced"
                args.dataset = "SampleDataset"
                args.num_class = 20
                args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
                args.max_seqlen = 320

                if ood_dataset is None:
                    ood_dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)
                ood_iou, ood_dmap, ood_mAP_Avg_ALL = ood_test(
                    ood_dataset,
                    args,
                    model,
                    device,
                    class_mapping=args.class_mapping,
                    save_activation=False,
                    itr=itr,
                )
                
                args.dataset_name = "ActivityNet1.2"
                args.dataset = "AntSampleDataset"
                args.num_class = 100
                args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
                args.max_seqlen = 60

                # cond = np.mean(ood_dmap) > np.mean(ood_max_map)
                if cond:
                    ood_max_map = ood_dmap
                print(
                "||".join(
                    [
                        "MAX map @ {} = {:.3f} ".format(ood_iou[i], ood_max_map[i] * 100)
                        for i in range(len(ood_iou))
                    ]
                    )
                )
                ood_max_map = np.array(ood_max_map)
                print('mAP Avg 0.1-0.7: {}'.format(np.mean(ood_max_map[:7]) * 100))

if __name__ == '__main__':
    args = options.parser.parse_args()
    world_size = 4  # Set the number of GPUs
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)