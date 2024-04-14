import os
from collections import defaultdict
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import models
import options
import proposal_methods as PM
import utils.wsad_utils as utils
from ood_test import ood_test
import wsad_dataset
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def IND_dataframe_to_json(
    df,
    output_file,
    args,
):
    class_mapping=args.class_mapping
    if "ActivityNet" in args.dataset_name:
        videoname_file=os.path.join(args.path_dataset, args.dataset_name+"-Annotations/videoname.npy")
        url_file=os.path.join(args.path_dataset, args.dataset_name+"-Annotations/url.npy")
        df = df.copy()

        videoname_list = np.load(videoname_file)
        url_list = np.load(url_file)
        url_list = [url.decode("utf-8").split("?")[-1][2:] for url in url_list]
        videoname_mapper = {k.decode("utf-8"): v for k, v in zip(videoname_list, url_list)}

        with open(class_mapping, "r") as file:
            class_mapping = json.load(file)
            a_index2name = {
                int(k): v["anet name"] for k, v in class_mapping.items()
            }

        df["t-start"] = df["t-start"] * 16 / 25
        df["t-end"] = df["t-end"] * 16 / 25

        results = {}
        for index, row in df.iterrows():
            video_name = row["video-id"]
            video_id = videoname_mapper[video_name]
            label = row["label"]
            if int(label) not in a_index2name:
                continue
            entry = {
                "label": a_index2name[label],
                "score": row["score"],
                "segment": [row["t-start"], row["t-end"]],
            }

            if video_name in results:
                results[video_name].append(entry)
            else:
                results[video_name] = [entry]

    elif "Thumos" in args.dataset_name:
        videoname_file=os.path.join(args.path_dataset, args.dataset_name+"-Annotations/videoname.npy")
        df = df.copy()

        videoname_list = np.load(videoname_file)
        videoname_mapper = [k.decode("utf-8") for k in videoname_list]

        with open(class_mapping, "r") as file:
            class_mapping = json.load(file)

            t_index2name = {
                int(k): v["thu name"] for k, v in class_mapping.items()
            }

        df["t-start"] = df["t-start"] * 16 / 25
        df["t-end"] = df["t-end"] * 16 / 25

        results = {}
        for index, row in df.iterrows():
            video_id = row["video-id"]
            # video_id = videoname_mapper[video_id]
            label = int(row["label"])
            if label not in t_index2name:
                continue
            entry = {
                "label": t_index2name[label],
                "score": row["score"],
                "segment": [row["t-start"], row["t-end"]],
            }

            if video_id in results:
                results[video_id].append(entry)
            else:
                results[video_id] = [entry]
    
    else:
        raise Exception("No Such Dataset!!!")


    with open(output_file, "w") as file:
        json.dump({"results": results}, file)

def snippet_mAP(results, gt, class_mapping=None):

    Acc = []
    Attn = []
    print(len(results))

    for vname in results.keys():
        cas = results[vname]["cas"]
        cas = torch.argmax(cas, dim=-1, keepdim=True)

        attn = results[vname]["attn"]
        proposals = gt[gt["video-id"] == vname]

        gt_p = torch.zeros_like(cas) + 100

        for index, row in proposals.iterrows():
            start = row["t-start"]
            end = row["t-end"]
            label = row["label"]
            if start >= cas.size(1):
                continue

            if end >= cas.size(1):
                end = cas.size(1)-1
            
            for t in range(start, end+1):
                gt_p[:,t,0] = label
        
        for t in range(cas.size(1)):
            if gt_p[:,t,0] == 100:
                continue
            if cas[:,t,0] == gt_p[:,t,0]:
                Acc.append(1)
            else:
                Acc.append(0)
            Attn.append(attn[:,t,0])
    
    print(sum(Acc)/len(Acc)*100)

    # Define the number of bins or intervals
    num_bins = 10

    # Initialize lists to store accuracy for each interval
    accuracy_by_bin = []

    # Initialize the bin edges
    bin_edges = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    print(bin_edges)

    # Calculate accuracy in each bin
    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        in_bin = [Acc[j] for j in range(len(Attn)) if lower_bound <= Attn[j] < upper_bound]
        accuracy = sum(in_bin) / len(in_bin) * 100 if len(in_bin) > 0 else 0
        accuracy_by_bin.append(accuracy)
    print(accuracy_by_bin)

    exit()

# "video_test_0000814": {"subset": "test", "annotations": [{"segment": ["59.6", "64.2"], "label": "ThrowDiscus"}, 
# {"segment": ["116.9", "120.2"], "label": "ThrowDiscus"}, {"segment": ["154.8", "160.4"], "label": "ThrowDiscus"}, 
# {"segment": ["233.6", "240.1"], "label": "ThrowDiscus"}, {"segment": ["305.5", "321.1"], "label": "ThrowDiscus"}, 
# {"segment": ["331.7", "337.1"], "label": "ThrowDiscus"}, {"segment": ["21.5", "24.1"], "label": "ThrowDiscus"}, 
# {"segment": ["24.8", "29.8"], "label": "ThrowDiscus"}, {"segment": ["32.2", "34.8"], "label": "ThrowDiscus"}, 
# {"segment": ["78.7", "85.6"], "label": "ThrowDiscus"}, {"segment": ["165.7", "168.3"], "label": "ThrowDiscus"}]},



@torch.no_grad()
def test(itr, dataset, args, model, device, save_activation=False, ind_class_mapping=False, snippet_classification=False):
    model.eval()
    done = False
    instance_logits_stack = []
    labels_stack = []

    if "ActivityNet" in args.dataset_name and ind_class_mapping: #Thumos->Anet
        if "1.2" in args.dataset_name:
            class_mapping = json.load(open("class_mapping/t2a_class_mapping.json", "r"))
            target_class_names = [v["anet name"] for v in class_mapping.values()]
        else:
            class_mapping = json.load(open("class_mapping/t2a_plus_class_mapping.json", "r"))
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
    elif "Thumos" in args.dataset_name and ind_class_mapping:  # Anet->Thumos
        class_mapping = json.load(open("class_mapping/a2t_class_mapping.json", "r"))
        target_class_names = [v["thu name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["thu idx"]) for item in class_mapping.values()
        ]

    proposals = []
    results = defaultdict(dict)
    while not done:
        if dataset.currenttestidx % (len(dataset.testidx) // 10) == 0:
            print('Testing test data point %d of %d' % (dataset.currenttestidx, len(dataset.testidx)))

        if ind_class_mapping:
            features, labels, vn, done, label_names = dataset.load_data(
                is_training=False, return_label_names=True
            )
            # skip if the label is not in the target class names
            keep = True
            for label_name in label_names:
                if label_name not in target_class_names:
                    keep = False
                    break
            if not keep:
                # continue
                pass
        else:
            features, labels, vn, done = dataset.load_data(is_training=False)

        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(Variable(features), is_training=False, seq_len=seq_len, opt=args)
            element_logits = outputs['cas']
            results[vn.decode("utf-8")] = {'cas': outputs['cas'], 'attn': outputs['attn']}
            proposals.append(getattr(PM, args.proposal_method)(vn, outputs))
            if isinstance(element_logits, list):
                element_logits = torch.stack(element_logits, dim=0).mean(dim=0)
            logits = element_logits.squeeze(0)
        tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features) / 8)), dim=0)[0], dim=0),
                        dim=0).cpu().data.numpy()

        ''' 
        atn=outputs["attn"][0,:,0].cpu()
        back=outputs["cas"].softmax(-1)[..., [-1]][0,:,0].cpu()
        uct=outputs["uct"][:,0].cpu()

        atn = np.array(atn)
        back = np.array(back)
        uct = np.array(uct)
        plt.plot(range(len(atn)), 1-atn, c="r", label="1-atn")
        plt.plot(range(len(back)), back, c="g", label="cas[:-1]")
        plt.plot(range(len(uct)), uct, c="b", label="uct(cas[:20])")

        plt.title(vn.decode("utf-8"))
        plt.legend(loc="lower right")
        plt.savefig("/data0/lixunsong/liuyangcen/CVPR2024/uct/"+vn.decode("utf-8")+".jpg")
        plt.clf()
        '''

        instance_logits_stack.append(tmp)
        labels_stack.append(labels)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    proposals = pd.concat(proposals).reset_index(drop=True)

    if save_activation:

        with open(
            "proposal_results/IND_activation.pkl", "wb"
        ) as file:
            pickle.dump(results, file) # 'video_test_0001558': {'cas': tensor(), 'attn': tensor()}

        IND_dataframe_to_json(
            proposals,
            "proposal_results/IND_proposals.json",
            args,
        )

    # CVPR2020
    if 'Thumos14' in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if ind_class_mapping:
            dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, selected_class_indices=target_class_indices)
        else:
            dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        dmap_detect.prediction = proposals
        if snippet_classification: # classification
            smap = snippet_mAP(results, dmap_detect.ground_truth)
        dmap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        if ind_class_mapping:
            dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='validation', selected_class_indices=target_class_indices)
        else:
            dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='validation')
        dmap_detect.prediction = proposals
        if snippet_classification: # classification
            smap = snippet_mAP(results, dmap_detect.ground_truth)
        dmap = dmap_detect.evaluate() # localization

    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' % cmap)
    print('||'.join(['map @ {} = {:.3f} '.format(iou[i], dmap[i] * 100) for i in range(len(iou))]))
    
    if 'Thumos14' in args.dataset_name:
        print('mAP 0.1-0.7: {:.3f}'.format(sum(dmap[:7]) / 7 * 100))
    else:
        print('mAP 0.5-0.95: {:.3f}'.format(sum(dmap) / len(iou) * 100))

    return iou, dmap

def main_thumos2anet(ind_class_mapping):
    print("Thumos14 -> ActivityNet1.2")

    args = options.parser.parse_args()
    device = torch.device("cuda")


    # Thumos 14
    args.ckpt_path = "ckpt/best_delu_thumos.pkl"
    # args.ckpt_path = "ckpt/best_delu_adapter.pkl"
    # args.ckpt_path = 'ckpt/best_ddg_thumos.pkl'
    # args.AWM = 'DDG_Net'

    args.proposal_method = 'multiple_threshold_hamnet'

    args.dataset_name = "Thumos14reduced"
    args.dataset = "SampleDataset"
    args.num_class = 20
    args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
    args.max_seqlen = 320
    args.scales = [1]
    args.class_mapping = "class_mapping/t2a_class_mapping.json"
    args.use_model = "DELU"

    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)

    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=True)

    iou, dmap = test(-1, dataset, args, model, device, save_activation=True, ind_class_mapping=ind_class_mapping)
    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    print('Thumos14: mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5]) * 100,
                                                                             np.mean(dmap[:7]) * 100,
                                                                             np.mean(dmap) * 100))
    exit()
    # Anet
    args.dataset_name = "ActivityNet1.2"
    args.dataset = "AntSampleDataset"
    args.num_class = 100
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
    args.max_seqlen = 60
    # args.scales = [1] # [1, 3, 7, 15]
    args.mapping = "class_mapping/t2a_class_mapping.json"

    # model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    # model.load_state_dict(torch.load(args.ckpt_path), strict=False)
    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)

    iou, dmap, mAP_Avg_ALL = ood_test(
                    dataset,
                    args,
                    model,
                    device,
                    class_mapping=args.class_mapping,
                    save_activation=True,
                )

    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    ood_max_map = np.array(dmap)
    print('ActivityNet1.2: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))


def main_anet2thumos(ind_class_mapping):
    print("ActivityNet1.2 -> Thumos14")

    args = options.parser.parse_args()
    device = torch.device("cuda")

    # Anet
    args.class_mapping = "class_mapping/a2t_class_mapping.json"
    args.ckpt_path = "ckpt/best_delu_adapter_act.pkl" # "ckpt/best_delu_act.pkl" # "ckpt/best_ddg_act.pkl" # "ckpt/best_base_act.pkl"
    # args.AWM = 'DDG_Net'
    args.proposal_method = 'multiple_threshold_hamnet'

    args.dataset_name = "ActivityNet1.2"
    args.dataset = "AntSampleDataset"
    args.num_class = 100
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
    args.max_seqlen = 60
    args.scales = [13] # 13

    args.use_model = "DELU"
    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)
    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)
    '''
    iou, dmap = test(-1, dataset, args, model, device, save_activation=True, ind_class_mapping=ind_class_mapping)
    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    ood_max_map = np.array(dmap)
    print('ActivityNet1.2: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))
    '''
    # Thumos 14
    args.dataset_name = "Thumos14reduced"
    args.dataset = "SampleDataset"
    args.num_class = 20
    args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
    args.max_seqlen = 320

    # model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    # model.load_state_dict(torch.load(args.ckpt_path), strict=False)

    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)

    iou, dmap, mAP_Avg_ALL = ood_test(
                    dataset,
                    args,
                    model,
                    device,
                    class_mapping=args.class_mapping,
                    save_activation=True
                )
    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    print('Thumos14: mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5]) * 100,
                                                                             np.mean(dmap[:7]) * 100,
                                                                             np.mean(dmap) * 100))
    
def main_thumos2hacs(ind_class_mapping):

    print("Thumos14 -> HACS")

    args = options.parser.parse_args()
    device = torch.device("cuda")


    # Thumos 14
    # args.ckpt_path = "/data0/lixunsong/liuyangcen/ECCV2022-DELU/work_dir/thumos_delu_pgd/ckpt/best_DELU.pkl" # "ckpt/best_base_thumos.pkl"

    args.ckpt_path = "ckpt/best_ddg_thumos.pkl"
    args.AWM = 'DDG_Net'
    args.proposal_method = 'multiple_threshold_hamnet'
    args.use_model = "DELU_DDG"

    args.dataset_name = "Thumos14reduced"
    args.dataset = "SampleDataset"
    args.num_class = 20
    args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
    args.max_seqlen = 320
    args.scales = [1]
    args.class_mapping = "class_mapping/t2a_plus_class_mapping.json"
    
    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)

    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)
    

    iou, dmap = test(-1, dataset, args, model, device, save_activation=True, ind_class_mapping=ind_class_mapping)
    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    print('Thumos14: mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5]) * 100,
                                                                             np.mean(dmap[:7]) * 100,
                                                                             np.mean(dmap) * 100))

    # Anet
    args.dataset_name = "ActivityNet1.3"
    args.dataset = "AntPlusSampleDataset"
    args.num_class = 9
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.3"
    args.max_seqlen = 100
    args.scales = [1]

    # model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    # model.load_state_dict(torch.load(args.ckpt_path), strict=False)

    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)

    iou, dmap, mAP_Avg_ALL = ood_test(
                    dataset,
                    args,
                    model,
                    device,
                    class_mapping=args.class_mapping,
                    save_activation=True,
                )

    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    ood_max_map = np.array(dmap)
    print('HACS: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))


def main_anet2hacs(ind_class_mapping):

    print("ActivityNet1.2 -> HACS")

    args = options.parser.parse_args()
    device = torch.device("cuda")


    # Anet
    args.class_mapping = "class_mapping/a2a_plus_class_mapping.json"
    args.ckpt_path = "ckpt/best_ddg_act.pkl"

    args.AWM = 'DDG_Net'
    args.proposal_method = 'multiple_threshold_hamnet'
    args.use_model = "DELU_DDG_ACT"

    args.dataset_name = "ActivityNet1.2"
    args.dataset = "AntSampleDataset"
    args.num_class = 100
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
    args.max_seqlen = 60
    args.scales = [7] # 13

    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)
    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)   
    '''
    iou, dmap = test(-1, dataset, args, model, device, save_activation=True, ind_class_mapping=ind_class_mapping)

    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    ood_max_map = np.array(dmap)
    print('ActivityNet1.2: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))
    '''
    # Anet
    args.dataset_name = "ActivityNet1.3"
    args.dataset = "AntPlusSampleDataset"
    args.num_class = 9
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.3"
    args.max_seqlen = 300
    args.scales = [13]

    # model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    # model.load_state_dict(torch.load(args.ckpt_path), strict=False)

    dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)

    iou, dmap, mAP_Avg_ALL = ood_test(
                    dataset,
                    args,
                    model,
                    device,
                    class_mapping=args.class_mapping,
                    save_activation=True,
                )

    print(
    "||".join(
        [
            "MAX map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
            for i in range(len(iou))
        ]
        )
    )
    ood_max_map = np.array(dmap)
    print('HACS: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))


if __name__ == '__main__':
    ind_class_mapping = False
    main_thumos2anet(ind_class_mapping)
    # main_anet2thumos(ind_class_mapping)
    # main_thumos2hacs(ind_class_mapping)
    # main_anet2hacs(ind_class_mapping)
