import json
import pickle
from pathlib import Path
from collections import defaultdict

from rich import print
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.optim as optim
import torch.nn.functional as F

import models
import options
import wsad_dataset
import single_stream_model
import proposal_methods as PM
import utils.wsad_utils as utils
from eval.eval_detection import ANETdetection
from eval.classificationMAP import getClassificationMAP as cmAP

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def dataframe_to_json(
    df,
    output_file,
    videoname_file="/data0/lixunsong/Datasets/ActivityNet1.2/ActivityNet1.2-Annotations/videoname.npy",
    url_file="/data0/lixunsong/Datasets/ActivityNet1.2/ActivityNet1.2-Annotations/url.npy",
    t2a_class_mapping="t2a_class_mapping.json",
):
    df = df.copy()

    videoname_list = np.load(videoname_file)
    url_list = np.load(url_file)
    url_list = [url.decode("utf-8").split("?")[-1][2:] for url in url_list]
    videoname_mapper = {k.decode("utf-8"): v for k, v in zip(videoname_list, url_list)}

    with open(t2a_class_mapping, "r") as file:
        t2a_class_mapping = json.load(file)
        a_index2name = {
            int(v["anet idx"]): v["anet name"] for k, v in t2a_class_mapping.items()
        }

    df["t-start"] = df["t-start"] * 16 / 25
    df["t-end"] = df["t-end"] * 16 / 25

    results = {}
    for index, row in df.iterrows():
        video_id = row["video-id"]
        video_id = videoname_mapper[video_id]
        label = row["label"]
        entry = {
            "label": a_index2name[label],
            "score": row["score"],
            "segment": [row["t-start"], row["t-end"]],
        }

        if video_id in results:
            results[video_id].append(entry)
        else:
            results[video_id] = [entry]

    with open(output_file, "w") as file:
        json.dump({"results": results}, file)


def convert_one_hot_label(label, class_mapper, num_classes):
    nonzeros = label.nonzero()
    video_indices = list(nonzeros[0])
    class_indices = list(nonzeros[1])
    nonzeros = [list(idx) for idx in nonzeros]
    new_label = np.zeros((label.shape[0], num_classes))
    for vid, cid in zip(video_indices, class_indices):
        if cid in class_mapper.keys():
            new_label[vid, class_mapper[cid]] = 1
        else:
            print("class {} of video {} not in class mapper".format(cid, vid))
    return new_label


@torch.no_grad()
def ood_test(
    dataset,
    args,
    model,
    device,
    class_mapping=None,
    save_activation=True,
    itr=-1,
    class_mapping_param=None,
    threshold=0.9 # threshold for pseudo label
):
    if class_mapping and class_mapping_param==None: # 走的上分支
        class_mapping = json.load(open(class_mapping, "r"))

        if "ActivityNet" in args.dataset_name: #Thumos->Anet
            target_class_names = [v["anet name"] for v in class_mapping.values()]
            target_class_indices = [
                int(item["anet idx"]) for item in class_mapping.values()
            ]
            source_class_indices = [int(k) for k in class_mapping.keys()] # [3, 10, 11, 12, 13, 14, 15, 18]
            idx_mapping = {int(k): int(v["anet idx"]) for k, v in class_mapping.items()} # Thumos->Anet
            reversed_idx_mapping = {
                int(v["anet idx"]): int(k) for k, v in class_mapping.items()
            }
        elif "Thumos" in args.dataset_name:  # Anet->Thumos
            target_class_names = [v["thu name"] for v in class_mapping.values()]
            target_class_indices = [
                int(item["thu idx"]) for item in class_mapping.values()
            ]
            source_class_indices = [int(k) for k in class_mapping.keys()] # [3, 10, 11, 12, 13, 14, 15, 18]
            idx_mapping = {int(k): int(v["thu idx"]) for k, v in class_mapping.items()} # Thumos->Anet
            reversed_idx_mapping = {
                int(v["thu idx"]): int(k) for k, v in class_mapping.items()
            }
    else:
        if class_mapping_param==None:
            raise Exception("no class mapping parameters!")
        target_class_names = class_mapping_param["target_class_names"]
        target_class_indices = class_mapping_param["target_class_indices"]
        source_class_indices = class_mapping_param["source_class_indices"]
        idx_mapping = class_mapping_param["idx_mapping"]
        reversed_idx_mapping = class_mapping_param["reversed_idx_mapping"]

    model.eval()
    done = False
    instance_logits_stack = []
    labels_stack = []

    proposals = []
    results = defaultdict(dict)

    while not done:
        if dataset.currenttestidx % (len(dataset.testidx) // 5) == 0:
            print(
                "Testing test data point %d of %d"
                % (dataset.currenttestidx, len(dataset.testidx))
            )

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
            continue
        '''
        if args.modality == "rgb":
            features = features[:, :1024]
        elif args.modality == "flow":
            features = features[:, 1024:]
        elif args.modality == "fusion":
            pass
        else:
            raise NotImplementedError
        '''
            
        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(features, is_training=False, seq_len=seq_len, opt=args, ood=True)
            element_logits = outputs['cas']
            pred_proposals = getattr(PM, args.proposal_method)(vn, outputs) # multiple_threshold_hamnet
            proposals.append(pred_proposals)
            if isinstance(element_logits, list):
                element_logits = torch.stack(element_logits, dim=0).mean(dim=0)
            logits = element_logits.squeeze(0).detach().cpu()

        tmp = (
            F.softmax(
                torch.mean(
                    torch.topk(logits, k=int(np.ceil(len(features) / 8)), dim=0)[0],
                    dim=0,
                ),
                dim=0,
            )
            .cpu()
            .data.numpy()
        )

        instance_logits_stack.append(tmp) # predicted label, numpy 21, threshold
        labels_stack.append(labels) # gt label, numpy 20
        

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    proposals = pd.concat(proposals).reset_index(drop=True)
    proposals["label"] = proposals["label"].astype(int)
    proposals = proposals[proposals["label"].isin(source_class_indices)]
    proposals["label"] = proposals["label"].map(idx_mapping)

    results["proposals"] = proposals
    if save_activation:
        try:
            with open(
                Path(args.checkpoint).parents[1] / "thu2anet1.2_activation.pkl", "wb"
            ) as file:
                pickle.dump(results, file)

            dataframe_to_json(
                proposals,
                Path(args.checkpoint).parents[1] / "thu2anet1.2_proposals.json",
            )
        except:
            pass

    # CVPR2020
    if "Thumos14" in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        dmap_detect = ANETdetection(
            dataset.path_to_annotations,
            iou,
            args=args,
            subset="validation",
            selected_class_indices=target_class_indices,
        )
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()

    '''
    if args.dataset_name == "Thumos14":
        test_set = sio.loadmat("test_set_meta.mat")["test_videos"][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]["background_video"] == "YES":
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])
    '''

    if "ActivityNet" in args.dataset_name:
        labels_stack = convert_one_hot_label(labels_stack, reversed_idx_mapping, 21)
    elif "Thumos14" in args.dataset_name:
        labels_stack = convert_one_hot_label(labels_stack, reversed_idx_mapping, 101)

    instance_logits_stack = np.take(instance_logits_stack, source_class_indices, axis=1) # 193,8
    labels_stack = np.take(labels_stack, source_class_indices, axis=1)

    cmap = cmAP(instance_logits_stack, labels_stack)

    print("OOD classification map %f" % cmap)
    print(
        "||".join(
            [
                "map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
                for i in range(len(iou))
            ]
        )
    )
    print("mAP Avg ALL: {:.3f}".format(sum(dmap) / len(iou) * 100))

    mAP_Avg_ALL = sum(dmap) / len(iou) * 100

    if itr == -1:
        results = {f"map@{iou}": mAP * 100 for iou, mAP in zip(iou, dmap)}
        results["map avg"] = sum(dmap) / len(iou) * 100
        results["cmap"] = cmap
        results["eval dataset"] = args.dataset_name
        output_path = (Path(args.checkpoint).parents[1] / "ood_results.json").as_posix()
        if Path(output_path).exists():
            with open(output_path, "r") as file:
                ood_results = json.load(file)
        else:
            ood_results = {}
        ood_results[args.checkpoint] = results
        with open(output_path, "w") as file:
            json.dump(ood_results, file)

    return iou, dmap, mAP_Avg_ALL

def ood_tta_test(dataset,
    args,
    model,
    device,
    class_mapping=None,
    save_activation=True,
    itr=-1,): # 这个模块用于直接进行tta的test, 操纵多个iter, 训练参数模仿的anet的train的参数

    if class_mapping:
        class_mapping = json.load(open(class_mapping, "r"))
        target_class_names = [v["anet name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
        source_class_indices = [int(k) for k in class_mapping.keys()] # [3, 10, 11, 12, 13, 14, 15, 18]
        idx_mapping = {int(k): int(v["anet idx"]) for k, v in class_mapping.items()}
        reversed_idx_mapping = {
            int(v["anet idx"]): int(k) for k, v in class_mapping.items()
        }
        
        class_mapping_param = {}
        class_mapping_param["target_class_names"] = target_class_names
        class_mapping_param["target_class_indices"] = target_class_indices
        class_mapping_param["source_class_indices"] = source_class_indices
        class_mapping_param["idx_mapping"] = idx_mapping
        class_mapping_param["reversed_idx_mapping"] = reversed_idx_mapping

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # save_path = Path(args.dir_path) / "checkpoints"
    prev_last_path, prev_best_pah = None, None
    iou, dmap = None, None

    max_mAP_Avg_ALL = 0
    for itr in range(args.max_iter): # 先ood_test再用上一轮的伪标签train, 这里的问题是threshold不好确定而且可能是动态的
        iou, dmap, mAP_Avg_ALL = ood_test(dataset, args, model, device, class_mapping=True, class_mapping_param=class_mapping_param)
        max_mAP_Avg_ALL = max(mAP_Avg_ALL, max_mAP_Avg_ALL)
        print("max_mAP_Avg_ALL: ", max_mAP_Avg_ALL)
        dataset.update_pseudoidx()
        print("training with: ", len(dataset.pseudoidx))
        temp_loss = train(itr, dataset, args, model, device, optimizer)
        print(iter, temp_loss)
        dataset.pseudo_multihot *= 0

if __name__ == "__main__":
    args = options.parser.parse_args()
    device = torch.device("cuda")


    if "thumos" in args.dataset_name.lower():  # only support thumos14 for now
        args.dataset_name = "ActivityNet1.2"
        args.dataset = "AntSampleDataset"
        args.num_class = 100
        args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
        args.max_seqlen = 60

    if "activitynet" in args.dataset_name.lower():  # only support thumos14 for now

        args.dataset_name = "Thumos14reduced"
        args.dataset = "SampleDataset"
        args.num_class = 20
        args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14"
        args.max_seqlen = 320

        dataset = getattr(wsad_dataset, args.dataset)(args)

        model = getattr(models, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
        model.load_state_dict(torch.load(args.ckpt_path))

        iou, dmap = ood_test(-1, dataset, args, model, device)
        print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5]) * 100,
                                                                                np.mean(dmap[:7]) * 100,
                                                                                np.mean(dmap) * 100))
