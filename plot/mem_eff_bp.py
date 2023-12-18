import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from utils import BLOCK_SIZE_TO_COLOR, DATASETS_META, MODELS_META, RESULTS_DIR

def usg_from_analysis(analysis, block_size, num_gpu_blocks):
    physical_blocks = set()
    total_physical_blocks = 0
    total_tokens = 0
    for seq_group in analysis:
        for seq in analysis[seq_group]:
            for pb_number, tokens in analysis[seq_group][seq]["physical"].items():
                if pb_number not in physical_blocks:
                    physical_blocks.add(pb_number)
                    total_physical_blocks += 1
                    total_tokens += tokens

    real_usg = total_tokens / (block_size * num_gpu_blocks)
    used_only_mem_usg = total_tokens / (block_size * total_physical_blocks)

    return real_usg, used_only_mem_usg

def plot_mem_efficiency_bp(
    saved_dir, dataset, model, block_sizes, req_rate, seed, duration
):
    mem_utils = []
    mem_utils_ = []
    for block_size in block_sizes:
        input_dir = os.path.join(RESULTS_DIR, dataset, model, "n1", f"block{block_size}",
                                f"req-rate-{req_rate}", f"seed{seed}", f"duration-{duration}"
                                )
        
        stats_path = os.path.join(input_dir, "stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        analysis_path = os.path.join(input_dir, "analysis.pkl")
        with open(analysis_path, "rb") as f:
            analysis = pickle.load(f)

        usage = []
        usage_ = []
        for step in analysis:
            usg, usg_ = usg_from_analysis(step["analysis"], block_size, stats["num_gpu_blocks"])
            usage.append(usg)
            usage_.append(usg_)
        
        mem_util = np.mean(usage)
        mem_util_ = np.mean(usage_)

        mem_utils.append(mem_util)
        mem_utils_.append(mem_util_)

    # Plot Total Memory Utiilzation
    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    colors = [color for color in BLOCK_SIZE_TO_COLOR.values()]
    labels = [label for label in BLOCK_SIZE_TO_COLOR.keys()]

    x = np.linspace(1, len(block_sizes), len(block_sizes))
    ax.bar(x, mem_utils, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=18)

    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_ylim(top=1.05)

    ax.set_xlabel("Block Size", size=18)
    ax.set_ylabel("Memory Utilization", size=18)

    dataset_name = DATASETS_META[dataset]["name"]
    model_name = MODELS_META[model]["name"]
    ax.set_title(f"{dataset_name} | {model_name} | {req_rate}", size=20)

    filename = f"total_mem_eff_{dataset}_{model}_{req_rate}_{seed}_{duration}.pdf"
    output_path = os.path.join(saved_dir, filename)

    print(f"Saving {filename}...")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print("Finished.")

    # Plot Allocated Memory Utilization
    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    colors = [color for color in BLOCK_SIZE_TO_COLOR.values()]
    labels = [label for label in BLOCK_SIZE_TO_COLOR.keys()]

    x = np.linspace(1, len(block_sizes), len(block_sizes))
    ax.bar(x, mem_utils_, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=18)

    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_ylim(top=1.05)

    ax.set_xlabel("Block Size", size=18)
    ax.set_ylabel("Allocated Memory Utilization", size=18)

    dataset_name = DATASETS_META[dataset]["name"]
    model_name = MODELS_META[model]["name"]
    ax.set_title(f"{dataset_name} | {model_name} | {req_rate}", size=20)

    filename = f"mem_eff_{dataset}_{model}_{req_rate}_{seed}_{duration}.pdf"
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
    args = parser.parse_args()

    save_dir_abs_path = os.path.abspath(args.save_dir)
    block_sizes = [int(blk_size) for blk_size in args.block_sizes.split(",")]

    plot_mem_efficiency_bp(save_dir_abs_path, args.dataset, args.model, block_sizes,
                        args.req_rate, args.seed, args.duration)
    
# python plot/mem_eff_bp.py ./figures/mem_eff_bp/ --dataset alpaca \
# --block-sizes 1,2,4,8,16,32,64,128,256 --model opt-13b --req-rate 31.5