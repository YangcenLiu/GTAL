import json
import pickle
from pathlib import Path
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import options
import wsad_dataset
from eval.eval_detection import ANETdetection


def interpolate_activation(curve, duration, interp_ratio=32):
    x = np.arange(0, curve_1.shape[0])
    interp_curve_1 = interp1d(
        x, curve_1, kind="linear", axis=0, fill_value="extrapolate"
    )
    curve_1 = interp_curve_1(np.arange(0, curve_1.shape[0], 1 / interp_ratio))


def visualize(
    dataset,
    args,
    result_path,
    attn_path,
    compare_result_path,
    compare_attn_path,
    output_path="proposal_visualize",
    class_mapping: Optional[str] = None,
    relaxation: float = 10,
):
    if class_mapping and "Activity" in args.dataset_name:
        class_mapping = json.load(open(class_mapping, "r"))
        target_class_names = [v["anet name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
        source_class_indices = [int(k) for k in class_mapping.keys()]
        idx_mapping = {int(k): int(v["anet idx"]) for k, v in class_mapping.items()}
    elif class_mapping:
        class_mapping = json.load(open(class_mapping, "r"))
        target_class_names = [v["thu name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["thu idx"]) for item in class_mapping.values()
        ]
        source_class_indices = [int(k) for k in class_mapping.keys()]
        idx_mapping = {int(k): int(v["thu idx"]) for k, v in class_mapping.items()}
        # dataset.filter_by_class_names(target_classes_names)

    save_path = os.path.join(output_path, "ThumosIND1_IND5")

    if "Thumos" in args.dataset_name:
        dmap_detect = ANETdetection(
            dataset.path_to_annotations,
            None,
            args=args,
            # subset="validation",
            selected_class_indices=target_class_indices,
        )
    else:
        dmap_detect = ANETdetection(
            dataset.path_to_annotations,
            None,
            args=args,
            subset="validation",
            selected_class_indices=target_class_indices,
        )
    ground_truth = dmap_detect.ground_truth

    with open(result_path, "rb") as file:
        results = json.load(file)
    proposals = results["results"]

    with open(attn_path, 'rb') as file:
        attn = pickle.load(file)

    with open(compare_result_path, "rb") as file:
        compare_results = json.load(file)
    compare_proposals = compare_results["results"]

    with open(compare_attn_path, 'rb') as file:
        compare_attn = pickle.load(file)

    # get the unique video ids
    video_ids = ground_truth["video-id"].unique()

    done = False
    while not done:
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

        video_id = vn.decode("utf-8")
        duration = dmap_detect.video_duration_dict[video_id]
        print(video_id)
        ax = plt.subplot(2, 1, 1)

        plot_single(
            ground_truth,
            proposals,
            attn,
            video_id,
            ax,
            duration,
            relaxation,
            "Thumos IND-1",
        )

        ax = plt.subplot(2, 1, 2)

        plot_single(
            ground_truth,
            compare_proposals,
            compare_attn,
            video_id,
            ax,
            duration,
            relaxation,
            "Thumos IND-3",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{video_id}.png"), dpi=300)
        plt.close()


def plot_single(
    ground_truth, proposals, attn, video_id, ax, duration, relaxation, title,
):

    # get the rows for this video from ground truth
    video_df_gt = ground_truth[ground_truth["video-id"] == video_id]
    gt_labels = video_df_gt["label"].unique()  # get ground truth labels

    thumos_json = {"CleanAndJerk": 3, "HammerThrow":10,"HighJump":11,"JavelinThrow":12,"LongJump":13,
        "PoleVault":14,
        "Shotput":15,
        "ThrowDiscus":18}
    anet_json = {"Clean and jerk": 10,
                "Hammer throw": 29,
                "High jump": 31,
                "Javelin throw": 35,
                "Long jump": 38,
                "Pole vault": 64,
                "Shot put": 74,
                "Discus throw": 15,
    }

    # plot the start and end times as lines for each ground truth
    for _, row in video_df_gt.iterrows():
        start = row["t-start"] * 16 / 25
        end = row["t-end"] * 16 / 25
        ax.plot(
            [start, end], [-0.1, -0.1], color="g", marker="|", linewidth=3
        )  # Bolder line for ground truth

    if video_id in proposals.keys():
        # get the rows for this video from proposals
        video_df_proposals = proposals[video_id]
        # plot the start and end times as lines for each proposal
        for row in video_df_proposals:
            start = row["segment"][0] # * 16 / 25
            end = row["segment"][1] # * 16 / 25
            score = row["score"]
            score = min(score, 1)
            label = row["label"]

            linestyle = (
                "-" if thumos_json[label] not in gt_labels else "solid"
            )  # Dashed line if label not in ground truth labels

            ax.plot(
                [start, end], [score, score], color="r", marker="|", linestyle=linestyle
            )

        ax.set_title(f"{title} {video_id}")
    else:
        ax.set_title(f"{title} {video_id} (No proposals)")


    ax.set_xlabel("Time")
    ax.set_xlim(-relaxation, duration + relaxation)
    ax.axvline(0)
    ax.axvline(duration)
    ax.set_ylim(-0.2, 1.2)

    attention = attn[video_id]["attn"][0][0, :, 0].cpu().numpy()
    timescale = np.linspace(0, duration, len(attention), axis=0)  # original timescale
    if timescale.ndim > 1:
        timescale = timescale[:,0]

    new_timescale = np.linspace(
        0, duration, int(duration * 100), axis=0
    )  # new timescale, assuming duration is in seconds and we want a point every 0.01 second
    if new_timescale.ndim > 1:
        new_timescale = new_timescale[:,0]

    interpolated_activations = np.interp(new_timescale, timescale, attention)
    ax.plot(new_timescale, interpolated_activations, color="b")


if __name__ == "__main__":
    args = options.parser.parse_args()
    name = "thumos"

    if name == "anet":
        args.dataset_name = "ActivityNet1.2"
        args.dataset = "AntSampleDataset"
        args.num_class = 100
        args.path_dataset = "/data0/lixunsong/Datasets/ActivityNet1.2/"
        args.max_seqlen = 60
        args.class_mapping = "t2a_class_mapping.json"

        dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)
        visualize(
            dataset,
            args,
            result_path="proposal_results/A_IND_proposals.json",
            attn_path="proposal_results/A_IND_activation.pkl",
            compare_result_path="proposal_results/A__proposals.json",
            compare_attn_path="proposal_results/A13_OOD_activation.pkl",
            class_mapping=args.class_mapping,
        )
        

    else:
        args.dataset_name = "Thumos14reduced"
        args.dataset = "SampleDataset"
        args.num_class = 20
        args.path_dataset = "/data0/lixunsong/Datasets/THUMOS14/"
        args.max_seqlen = 320
        args.scales = [1]
        args.class_mapping = "a2t_class_mapping.json"

        dataset = getattr(wsad_dataset, args.dataset)(args, classwise_feature_mapping=False)
        visualize(
            dataset,
            args,
            result_path="proposal_results/T_IND_proposals.json",
            attn_path="proposal_results/T_IND_activation.pkl",
            compare_result_path="proposal_results/IND_proposals.json",
            compare_attn_path="proposal_results/IND_activation.pkl",
            class_mapping=args.class_mapping,
        )
