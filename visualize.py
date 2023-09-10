import json
import pickle
from pathlib import Path
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
    compare_result_path,
    class_mapping: Optional[str] = None,
    relaxation: float = 10,
):
    if class_mapping:
        class_mapping = json.load(open(class_mapping, "r"))
        target_class_names = [v["anet name"] for v in class_mapping.values()]
        target_class_indices = [
            int(item["anet idx"]) for item in class_mapping.values()
        ]
        source_class_indices = [int(k) for k in class_mapping.keys()]
        idx_mapping = {int(k): int(v["anet idx"]) for k, v in class_mapping.items()}
        # dataset.filter_by_class_names(target_classes_names)

    root_path = Path(result_path).parent
    save_path = root_path / "visualizations"
    save_path.mkdir(exist_ok=True)

    dmap_detect = ANETdetection(
        dataset.path_to_annotations,
        None,
        args=args,
        subset="validation",
        selected_class_indices=target_class_indices,
    )
    ground_truth = dmap_detect.ground_truth

    with open(result_path, "rb") as file:
        results = pickle.load(file)
    proposals = results["proposals"]

    with open(compare_result_path, "rb") as file:
        compare_results = pickle.load(file)
    compare_proposals = compare_results["proposals"]

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

        ax = plt.subplot(2, 1, 1)

        plot_single(
            ground_truth,
            proposals,
            results,
            video_id,
            ax,
            duration,
            relaxation,
            "OOD test results",
        )

        ax = plt.subplot(2, 1, 2)

        plot_single(
            ground_truth,
            compare_proposals,
            compare_results,
            video_id,
            ax,
            duration,
            relaxation,
            "IND test results",
        )

        plt.tight_layout()
        plt.savefig(save_path / f"{video_id}.png", dpi=300)
        plt.close()


def plot_single(
    ground_truth, proposals, results, video_id, ax, duration, relaxation, title=""
):
    # get the rows for this video from ground truth
    video_df_gt = ground_truth[ground_truth["video-id"] == video_id]
    gt_labels = video_df_gt["label"].unique()  # get ground truth labels

    # plot the start and end times as lines for each ground truth
    for _, row in video_df_gt.iterrows():
        start = row["t-start"] * 16 / 25
        end = row["t-end"] * 16 / 25
        ax.plot(
            [start, end], [-0.1, -0.1], color="g", marker="|", linewidth=3
        )  # Bolder line for ground truth

    if video_id in proposals["video-id"].values:
        # get the rows for this video from proposals
        video_df_proposals = proposals[proposals["video-id"] == video_id]

        # plot the start and end times as lines for each proposal
        for _, row in video_df_proposals.iterrows():
            start = row["t-start"] * 16 / 25
            end = row["t-end"] * 16 / 25
            score = row["score"]
            score = min(score, 1)
            label = row["label"]

            linestyle = (
                "--" if label not in gt_labels else "-"
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

    attention = results[video_id]["attn"][0, :, 0].numpy()
    timescale = np.linspace(0, duration, len(attention))  # original timescale
    new_timescale = np.linspace(
        0, duration, int(duration * 100)
    )  # new timescale, assuming duration is in seconds and we want a point every 0.01 second
    interpolated_activations = np.interp(new_timescale, timescale, attention)
    ax.plot(new_timescale, interpolated_activations, color="b")


if __name__ == "__main__":
    parser = options.get_parser()
    # parser.add_argument("--class_mapping", type=str, default="t2a_class_mapping.json")
    args = parser.parse_args()

    dataset = getattr(wsad_dataset, args.dataset)(args)
    visualize(
        dataset,
        args,
        result_path="work_dir/2023-08-19T14-46-52_THU_fusion_debug/activation.pkl",
        compare_result_path="work_dir/2023-08-19T14-46-52_THU_fusion_debug/activation.pkl",
        class_mapping=args.class_mapping,
    )
