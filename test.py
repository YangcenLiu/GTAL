import os
from collections import defaultdict
from pathlib import Path
import json
import pickle

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


@torch.no_grad()
def test(itr, dataset, args, model, device, save_activation=False, ind_class_mapping=False):
    model.eval()
    done = False
    instance_logits_stack = []
    labels_stack = []
    
    if "ActivityNet" in args.dataset_name and ind_class_mapping: #Thumos->Anet
        if "1.2" in args.dataset_name:
            class_mapping = json.load(open("class_mapping/t2a_class_mapping.json", "r"))
        else:
            class_mapping = json.load(open("class_mapping/t2a_plus_class_mapping.json", "r"))
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
    elif "Thumos" in args.dataset_name and ind_class_mapping:  # Anet->Thumos
        class_mapping = json.load(open("class_mapping/a2t_class_mapping.json", "r"))
        target_class_indices = [
            int(item["thu idx"]) for item in class_mapping.values()
        ]

    proposals = []
    results = defaultdict(dict)
    while not done:
        if dataset.currenttestidx % (len(dataset.testidx) // 5) == 0:
            print('Testing test data point %d of %d' % (dataset.currenttestidx, len(dataset.testidx)))

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

        instance_logits_stack.append(tmp)
        labels_stack.append(labels)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    proposals = pd.concat(proposals).reset_index(drop=True)

    if save_activation:

        with open(
            "proposal_results/IND_activation.pkl", "wb"
        ) as file:
            pickle.dump(results, file)

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
        dmap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        if ind_class_mapping:
            dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='validation', selected_class_indices=target_class_indices)
        else:
            dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='validation')
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()

    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' % cmap)
    print('||'.join(['map @ {} = {:.3f} '.format(iou[i], dmap[i] * 100) for i in range(len(iou))]))
    print('mAP Avg ALL: {:.3f}'.format(sum(dmap) / len(iou) * 100))

    return iou, dmap

def main_thumos2anet(ind_class_mapping):
    print("Thumos14 -> ActivityNet1.2")

    args = options.parser.parse_args()
    device = torch.device("cuda")


    # Thumos 14
    args.ckpt_path = "ckpt/best_delu_thumos.pkl"
    args.dataset_name = "Thumos14reduced"
    args.dataset = "SampleDataset"
    args.num_class = 20
    args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
    args.max_seqlen = 320
    args.scales = [-3]
    args.class_mapping = "class_mapping/t2a_class_mapping.json"
    
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
    args.dataset_name = "ActivityNet1.2"
    args.dataset = "AntSampleDataset"
    args.num_class = 100
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
    args.max_seqlen = 60
    args.scales = [13]
    args.mapping = "class_mapping/t2a_class_mapping.json"

    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)
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
    args.ckpt_path = "ckpt/best_delu_act.pkl"
    args.dataset_name = "ActivityNet1.2"
    args.dataset = "AntSampleDataset"
    args.num_class = 100
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
    args.max_seqlen = 60
    args.scales = [13] # 13

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
    ood_max_map = np.array(dmap)
    print('ActivityNet1.2: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))
    
    # Thumos 14
    args.dataset_name = "Thumos14reduced"
    args.dataset = "SampleDataset"
    args.num_class = 20
    args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
    args.max_seqlen = 320
    args.scales = [1]

    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)

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
    
def main_thumos2anet_plus(ind_class_mapping):

    print("Thumos14 -> ActivityNet1.3")

    args = options.parser.parse_args()
    device = torch.device("cuda")


    # Thumos 14
    args.ckpt_path = "ckpt/best_delu_thumos.pkl"
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
    args.max_seqlen = 60
    args.scales = [13]

    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)
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
    print('ActivityNet1.3: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))


def main_anet2anet_plus(ind_class_mapping):

    print("ActivityNet1.2 -> ActivityNet1.3")

    args = options.parser.parse_args()
    device = torch.device("cuda")


    # Anet
    args.class_mapping = "class_mapping/a2a_plus_class_mapping.json"
    args.ckpt_path = "ckpt/best_delu_act.pkl"
    args.dataset_name = "ActivityNet1.2"
    args.dataset = "AntSampleDataset"
    args.num_class = 100
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
    args.max_seqlen = 60
    args.scales = [13] # 13

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
    ood_max_map = np.array(dmap)
    print('ActivityNet1.2: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))

    # Anet
    args.dataset_name = "ActivityNet1.3"
    args.dataset = "AntPlusSampleDataset"
    args.num_class = 9
    args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.3"
    args.max_seqlen = 60
    args.scales = [13]

    model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)
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
    print('ActivityNet1.3: mAP Avg 0.5-0.95: {}'.format(np.mean(ood_max_map[:10]) * 100))


if __name__ == '__main__':
    ind_class_mapping = True # True
    # main_thumos2anet(ind_class_mapping)
    # main_anet2thumos(ind_class_mapping)
    # main_thumos2anet_plus(ind_class_mapping)
    main_anet2anet_plus(ind_class_mapping)
    

