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

def Thumos14Feature(itr, dataset, args, model, writer=None, class_mapping=None):
    if class_mapping:
        class_mapping = json.load(open(class_mapping, "r"))
        target_class_names = [v["anet name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
        source_class_indices = [int(k) for k in class_mapping.keys()]
        idx_mapping = {int(k): int(v["anet idx"]) for k, v in class_mapping.items()}

    model.eval()
    done = False
    instance_logits_stack = []
    labels_stack = []

    proposals = []
    results = defaultdict(dict)
    all_features = [] # all features
    all_labels = [] # background and foreground

    # CVPR2020
    if "Thumos14" in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        # dmap_detect.prediction = proposals
        # dmap = dmap_detect.evaluate()
        gt_proposals = dmap_detect.ground_truth.groupby("video-id")
        # print(gt_proposals.get_group('video_test_0000004'))

    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        dmap_detect = ANETdetection(
            dataset.path_to_annotations, iou, args=args, subset="validation"
        )
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()


    while not done:
        if dataset.currenttestidx % (len(dataset.testidx) // 5) == 0:
            print(
                "Testing test data point %d of %d"
                % (dataset.currenttestidx, len(dataset.testidx))
            )

        features, labels, vn, done = dataset.load_data(is_training=False)

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
        
        gf_labels = np.zeros(seq_len) # 0: background 1: foreground

        features = torch.from_numpy(features).float().to(args.device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(features, is_training=False, seq_len=seq_len, opt=args)
            results[vn.decode("utf-8")] = {
                "cas": outputs["cas"].detach().cpu(),
                "attn": outputs["attn"].detach().cpu(),
            }
            pr = getattr(PM, args.proposal_method)(vn, outputs, args)
            proposals.append(pr)
            logits = outputs["cas"].squeeze(0).detach().cpu() # len,num_class

        for i in range(len(labels)):
            if i not in [3,10,11,12,13,14,15,18]:
                continue
            if labels[i] !=0 :
                all_features.append(np.array(
                    torch.mean((outputs["feat"][0,:,:] * outputs["cas"][0,:,i:i+1]), dim=0).cpu()))
                all_labels.append(i)
        all_features.append(np.array(
                    torch.mean((outputs["feat"][0,:,:] * outputs["cas"][0,:,0:0+1]), dim=0).cpu()))
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
        ) # num_class
        
        features = np.array(outputs["feat"].cpu())[0]
        
        for index, row in gt_prs.iterrows():
            for t in range(row['t-start'], row['t-end']):
                if row['t-start']>len(gf_labels) or row['t-end']>len(gf_labels):
                    continue

                gf_labels[t] = row["label"] # set foreground as 1
        for t in range(seq_len[0]):
            if gf_labels[t] not in [0,3,10,11,12,13,14,15,18]:
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

    results["proposals"] = proposals
    try:
        with open(
            Path(args.checkpoint).parents[1] / "activation.pkl", "wb"
        ) as file:
            pickle.dump(results, file)
    except:
        pass

    '''
    if args.dataset_name == "Thumos14":
        test_set = sio.loadmat("test_set_meta.mat")["test_videos"][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]["background_video"] == "YES":
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])
    '''

    cmap = cmAP(instance_logits_stack, labels_stack)
    print("Classification map %f" % cmap)
    print(
        "||".join(
            [
                "map @ {} = {:.3f} ".format(iou[i], dmap[i] * 100)
                for i in range(len(iou))
            ]
        )
    )
    print("mAP Avg ALL: {:.3f}".format(sum(dmap) / len(iou) * 100))

    if writer is not None:
        for k, v in zip(iou, list(dmap)):
            writer.add_scalar("val/mAP@{}".format(k), v, itr)
        writer.add_scalar("val/avg_mAP", np.mean(dmap), itr)

    return iou, dmap

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(data.shape[0]):
        if int(label[i]) == 0:
            s0 = plt.scatter(data[i, 0], data[i, 1], c="red", s=0.2) 
        elif int(label[i]) == 3:
            s3 = plt.scatter(data[i, 0], data[i, 1], c="green", s=0.2)
        elif int(label[i]) == 10:
            s10 = plt.scatter(data[i, 0], data[i, 1], c="magenta", s=0.2)
        elif int(label[i]) == 11:
            s11 = plt.scatter(data[i, 0], data[i, 1], c="blue", s=0.2)
        elif int(label[i]) == 12:
            s12 = plt.scatter(data[i, 0], data[i, 1], c="purple", s=0.2)
        elif int(label[i]) == 13:
            s13 = plt.scatter(data[i, 0], data[i, 1], c="black", s=0.2)
        elif int(label[i]) == 14:
            s14 = plt.scatter(data[i, 0], data[i, 1], c="orange", s=0.2)
        elif int(label[i]) == 15:
            s15 = plt.scatter(data[i, 0], data[i, 1], c="grey", s=0.2)
        elif int(label[i]) == 18:
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
                         't-SNE embedding of video features for Thumos14')
    plt.savefig("demo.png")


if __name__ == '__main__':
    parser = options.get_parser()
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, default="ckpt/best_delu_thumos.pkl"
    )
    parser.add_argument("--class_mapping", type=str, default="t2a_class_mapping.json")
    parser.add_argument("--vis_num", type=int, default=300) # sample num for vis
    args = options.get_args(parser)

    dataset = getattr(wsad_dataset, args.dataset)(args)

    models = getattr(models, args.use_model)(
        dataset.feature_size, dataset.num_class, opt=args
    ).to(args.device)
    models.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    all_features, all_labels = Thumos14Feature(-1, dataset, args, models, class_mapping=args.class_mapping)
    print(len(all_features))
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    chosen = random.sample(list(range(len(all_features))),args.vis_num)
    # visualise(all_features[chosen], all_labels[chosen])
    np.save("Thumos14features.npy", all_features[chosen])
    np.save("Thumos14labels.npy", all_labels[chosen])

