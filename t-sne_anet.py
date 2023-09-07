import json
from collections import defaultdict
from pathlib import Path
import pickle
import random
from rich import print
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F

import models
import single_stream_model
import options
import proposal_methods as PM
import utils.wsad_utils as utils
import wsad_dataset
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection

from time import time
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

def dataframe_to_json(
    df,
    output_file,
    videoname_file="/data/home/lyc/Dataset/ActivityNet1.2/ActivityNet1.2-Annotations/videoname.npy",
    url_file="/data/home/lyc/Dataset/ActivityNet1.2/ActivityNet1.2-Annotations/url.npy",
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

@torch.no_grad()
def AnetFeature(dataset, args, model, device, class_mapping=None):
    if class_mapping:
        class_mapping = json.load(open(class_mapping, "r"))
        target_class_names = [v["anet name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
        source_class_indices = [int(k) for k in class_mapping.keys()]
        idx_mapping = {int(k): int(v["anet idx"]) for k, v in class_mapping.items()}
        # dataset.filter_by_class_names(target_classes_names)

    model.eval()
    done = False
    instance_logits_stack = []
    labels_stack = []

    proposals = []
    results = defaultdict(dict)
    all_features = [] # all features
    all_labels = [] # background and foreground

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
        gt_proposals = dmap_detect.ground_truth.groupby("video-id")


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

        gt_prs = gt_proposals.get_group(str(vn, encoding='utf-8')) # ground truth proposals for vn
        if args.modality == "rgb":
            features = features[:, :1024]
        elif args.modality == "flow":
            features = features[:, 1024:]
        elif args.modality == "fusion":
            pass
        else:
            raise NotImplementedError

        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(features, is_training=False, seq_len=seq_len, opt=args)
            results[vn.decode("utf-8")] = {
                "cas": outputs["cas"].detach().cpu(),
                "attn": outputs["attn"].detach().cpu(),
            }
            proposals.append(getattr(PM, args.proposal_method)(vn, outputs, args))
            logits = outputs["cas"].squeeze(0).detach().cpu()

        for i in range(len(labels)):
            if i not in [10,29,31,35,38,63,74,15]:
                continue
            if labels[i] !=0 :
                class_mapping = {10:3,29:10,31:11,35:12,38:13,63:14,74:15,15:18}
                c = class_mapping[i]
                all_features.append(np.array(
                    torch.mean((outputs["feat"][0,:,:] * outputs["cas"][0,:,c:(c+1)]), dim=0).cpu()))
                all_labels.append(i)
        all_features.append(np.array(
                    torch.mean((outputs["feat"][0,:,:] * outputs["cas"][0,:,0:1]), dim=0).cpu()))
        all_labels.append(0)

        '''
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
        # features = np.array(outputs["feat"].cpu())[0]

        gf_labels = np.zeros(seq_len) # 0: background 1: foreground
        
        
        for index, row in gt_prs.iterrows():
            for t in range(row['t-start'], row['t-end']):
                if row['t-start']>len(gf_labels) or row['t-end']>len(gf_labels):
                    continue
                gf_labels[t] = row["label"] # set foreground as 1
        for t in range(seq_len[0]):
            if gf_labels[t] not in [0,10,29,31,35,38,63,74,15]:
                continue
            all_features.append(features[t])
            all_labels.append(gf_labels[t])

        instance_logits_stack.append(tmp)
        labels_stack.append(labels)
        '''

    return all_features, all_labels

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    proposals = pd.concat(proposals).reset_index(drop=True)
    proposals["label"] = proposals["label"].astype(int)
    proposals = proposals[proposals["label"].isin(source_class_indices)]
    proposals["label"] = proposals["label"].map(idx_mapping)

    results["proposals"] = proposals
    try:
        with open(
            Path(args.checkpoint).parents[1] / "thu2anet1.2_activation.pkl", "wb"
        ) as file:
            pickle.dump(results, file)

        dataframe_to_json(
            proposals, Path(args.checkpoint).parents[1] / "thu2anet1.2_proposals.json"
        )
    except:
        pass


    # cmap = cmAP(instance_logits_stack, labels_stack)
    # print("Classification map %f" % cmap)
    print(
        "||".join(
            [
                "map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
                for i in range(len(iou))
            ]
        )
    )
    print("mAP Avg ALL: {:.3f}".format(sum(dmap) / len(iou) * 100))

    return iou, dmap

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(data.shape[0]):
        if int(label[i]) == 0:
            s0 = plt.scatter(data[i, 0], data[i, 1], c="red", s=0.2) 
        elif int(label[i]) == 10:
            s3 = plt.scatter(data[i, 0], data[i, 1], c="green", s=0.2)
        elif int(label[i]) == 29:
            s10 = plt.scatter(data[i, 0], data[i, 1], c="magenta", s=0.2)
        elif int(label[i]) == 31:
            s11 = plt.scatter(data[i, 0], data[i, 1], c="blue", s=0.2)
        elif int(label[i]) == 35:
            s12 = plt.scatter(data[i, 0], data[i, 1], c="purple", s=0.2)
        elif int(label[i]) == 38:
            s13 = plt.scatter(data[i, 0], data[i, 1], c="black", s=0.2)
        elif int(label[i]) == 63:
            s14 = plt.scatter(data[i, 0], data[i, 1], c="orange", s=0.2)
        elif int(label[i]) == 74:
            s15 = plt.scatter(data[i, 0], data[i, 1], c="grey", s=0.2)
        elif int(label[i]) == 15:
            s18 = plt.scatter(data[i, 0], data[i, 1], c="brown", s=0.2)

    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s0,s3,s10,s11,s12,s13,s14,s15,s18),('Background','CleanAndJerk','HammerThrow','HighJump',
    'JavelinThrow','LongJump','PoleVault','Shotput','ThrowDiscus') ,loc = 'best', prop = {'size':6})
    plt.title(title)
    return fig

def visualise(features, labels):
    # data, label, n_samples, n_features = get_data()

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    result = tsne.fit_transform(features)
    print('result.shape',result.shape)
    fig = plot_embedding(result, labels,
                         't-SNE embedding of video features for Anet1.2')
    plt.savefig("I3DAnet.png")


if __name__ == '__main__':
    parser = options.get_parser()
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, default="ckpt/best_delu_thumos.pkl"
    )
    parser.add_argument("--model_num_calsses", type=int, default=20)
    parser.add_argument("--class_mapping", type=str, default="t2a_class_mapping.json")
    parser.add_argument("--vis_num", type=int, default=387) # sample num for vis
    args = parser.parse_args()

    print("evluating checkpoint {}".format(args.checkpoint))

    device = torch.device("cuda")
    dataset = getattr(wsad_dataset, args.dataset)(args)

    feature_size = (
        dataset.feature_size if args.modality == "fusion" else dataset.feature_size // 2
    )
    if args.modality == "fusion":
        models = getattr(models, args.use_model)(
            feature_size, args.model_num_calsses, opt=args
        ).to(device)
    else:
        models = single_stream_model.SingleDELU(
            feature_size, args.model_num_calsses, opt=args
        ).to(device)
    models.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    print("Model loaded from {}".format(args.checkpoint))

    all_features, all_labels = AnetFeature(dataset, args, models, device, class_mapping=args.class_mapping)

    print(len(all_features))
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    chosen = random.sample(list(range(len(all_features))),args.vis_num)
    visualise(all_features[chosen], all_labels[chosen])
    np.save("Anet1.2features.npy", all_features[chosen])
    np.save("Anet1.2labels.npy", all_labels[chosen])

