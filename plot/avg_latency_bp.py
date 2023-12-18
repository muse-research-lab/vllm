import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from utils import BLOCK_SIZE_TO_COLOR, DATASETS_META, MODELS_META, RESULTS_DIR

def plot_avg_latency_bp(
    saved_dir, dataset, model, block_sizes, req_rate, seed, duration, ylim = None
):
    latencies = []

    for block_size in block_sizes:
        input_path = os.path.join(RESULTS_DIR, dataset, model, "n1", f"block{block_size}",
                                  f"req-rate-{req_rate}", f"seed{seed}", f"duration-{duration}",
                                  "requests.pkl"
                                )

        with open(input_path, "rb") as f:
            requests = pickle.load(f)

        per_req_norm_latencies = []
        for req in requests:
            arrival_time = req["arrival_time"]
            finish_time = req["finish_time"]
            output_len = len(req["output"]["outputs"][0]["token_ids"])

            latency = finish_time - arrival_time
            norm_latency = latency / output_len
            per_req_norm_latencies.append(norm_latency)

        # Average normalized latency
        normalized_latency = np.mean(per_req_norm_latencies)

        latencies.append(normalized_latency)

    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    colors = [color for color in BLOCK_SIZE_TO_COLOR.values()]
    labels = [label for label in BLOCK_SIZE_TO_COLOR.keys()]

    x = np.linspace(1, len(block_sizes), 9)
    ax.bar(x, latencies, color=colors)

    
    if ylim:
        ax.set_ylim(top=ylim)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=18)

    ax.yaxis.set_tick_params(labelsize=18)

    base = ylim or ax.get_ylim()[1]
    for i in range(len(block_sizes)):
        if latencies[i] > base:
            offset = base - base / 10
            ax.text(x[i]-0.35, offset, str(latencies[i])[:4], color="white", size=12)

    ax.set_xlabel("Block Size", size=18)
    ax.set_ylabel("Average Latency (s/token)", size=18)

    dataset_name = DATASETS_META[dataset]["name"]
    model_name = MODELS_META[model]["name"]
    ax.set_title(f"{dataset_name} | {model_name} | {req_rate}", size=20)

    filename = f"avg_latency_{dataset}_{model}_{req_rate}_{seed}_{duration}.pdf"
    output_path = os.path.join(saved_dir, filename)

    print(f"Saving {filename}...")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--block-sizes", type=str, required=True)
    parser.add_argument("--req-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--ylim", type=float, default=None)
    args = parser.parse_args()

    save_dir_abs_path = os.path.abspath(args.save_dir)
    block_sizes = [int(blk_size) for blk_size in args.block_sizes.split(",")]

    plot_avg_latency_bp(save_dir_abs_path, args.dataset, args.model, block_sizes,
                        args.req_rate, args.seed, args.duration, args.ylim)
    
# python plot/avg_latency_bp.py ./figures/avg_latency_bp/ --dataset alpaca \
# --block-sizes 1,2,4,8,16,32,64,128,256 --model opt-13b --req-rate 31.5 \
# --ylim 0.5