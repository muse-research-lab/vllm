import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from utils import BLOCK_SIZE_TO_COLOR, DATASETS_META, MODELS_META, RESULTS_DIR

def plot_latency_breakdown_cdf(
    saved_dir, dataset, model, block_sizes, req_rate, seed, duration, xlim = None
):
    avg_running_lat = []
    avg_waiting_first_lat = []
    avg_waiting_again_lat= []

    full_running_lat = []
    full_waiting_first_lat = []
    full_waiting_again_lat= []

    for block_size in block_sizes:
        input_dir = os.path.join(RESULTS_DIR, dataset, model, "n1", f"block{block_size}",
                             f"req-rate-{req_rate}", f"seed{seed}", f"duration-{duration}"
                             )

        # Load Data
        request_path = os.path.join(input_dir, "requests.pkl")
        with open(request_path, "rb") as f:
            requests = pickle.load(f)

        stats_path = os.path.join(input_dir, "stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        status_matrix_path = os.path.join(input_dir, "status_matrix.pkl")
        with open(status_matrix_path, "rb") as f:
            status_matrix = pickle.load(f)

        # Process Data
        requests_dict = {}

        for req in requests:
            requests_dict[int(req["output"]["request_id"])] = req

        ts_adjusted = []
        for step in stats["steps"]:
            ts_adjusted.append(step["timestamp"]+stats["start_time"])

        running_lat = []
        waiting_first_lat = []
        waiting_again_lat = []

        previous_time = stats["start_time"]
        for i, row in enumerate(status_matrix):
            has_started = False
            waiting_before_running = False
            first_stop = False
            has_stopped = False

            running_acc = 0
            waiting_first_acc = 0
            waiting_again_cnt = 0
            
            for j, col in enumerate(row):
                current_time = ts_adjusted[j]

                if not has_started:
                    
                    # Waiting to start
                    if col == -1.0:
                        continue

                    # Found first start
                    elif col == 1.0:
                        has_started = True
                    else:
                        waiting_before_running = True
                        has_started = True

                else:
                    # Running without having stopped
                    if not first_stop:
                        if col == -1.0:
                            # Terminated without stopping
                            running_acc += current_time - previous_time
                            break

                        # Started or keeps running
                        elif col == 1.0:
                            if waiting_before_running:
                                waiting_before_running = False
                                waiting_first_acc += current_time - previous_time
                            else:
                                running_acc += current_time - previous_time
                        else:
                            if waiting_before_running:
                                waiting_first_acc += current_time - previous_time
                            else:
                                running_acc += current_time - previous_time
                                first_stop = True
                    
                    # Running after having stopped once
                    else:
                        if col == -1.0:
                            # Terminated without stopping
                            running_acc += current_time - previous_time
                            break

                        else:
                            if row[j-1] == 0:
                                waiting_again_cnt += current_time - previous_time
                            else:
                                running_acc += current_time - previous_time
                
                previous_time = current_time

            # Avoid warm up requests
            if running_acc == 0:
                continue

            output_len = len(requests_dict[i]["output"]["outputs"][0]["token_ids"])

            running_lat.append(running_acc / output_len)
            waiting_first_lat.append(waiting_first_acc / output_len)
            waiting_again_lat.append(waiting_again_cnt / output_len)
        
        avg_running_lat.append(np.mean(running_lat))
        avg_waiting_first_lat.append(np.mean(waiting_first_lat))
        avg_waiting_again_lat.append(np.mean(waiting_again_lat))

        full_running_lat.append(running_lat)
        full_waiting_first_lat.append(waiting_first_lat)
        full_waiting_again_lat.append(waiting_again_lat)

    # Plot Run Latency
    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    for block_size, lats in zip(block_sizes, full_running_lat):
        x = np.sort(lats)
        y = 1. * np.arange(len(lats)) / (len(lats) - 1)

        ax.plot(x, y, color=BLOCK_SIZE_TO_COLOR[str(block_size)], label=block_size)

    if xlim:
        ax.set_xlim(left=-0.02, right=xlim)

    ax.set_ylabel("Probability", size=18)
    ax.set_xlabel("Running Latency (s/token)", size=18)

    ax.legend(fontsize=16)

    dataset_name = DATASETS_META[dataset]["name"]
    model_name = MODELS_META[model]["name"]
    ax.set_title(f"{dataset_name} | {model_name} | {req_rate}", size=20)

    filename = f"run_latency_cdf_{dataset}_{model}_{req_rate}_{seed}_{duration}.pdf"
    output_path = os.path.join(saved_dir, filename)

    print(f"Saving {filename}...")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print("Finished.")

    # Plot First Wait Latency
    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    for block_size, lats in zip(block_sizes[:-2], full_waiting_first_lat[:-2]):
        x = np.sort(lats)
        y = 1. * np.arange(len(lats)) / (len(lats) - 1)

        ax.plot(x, y, color=BLOCK_SIZE_TO_COLOR[str(block_size)], label=block_size)

    if xlim:
        ax.set_xlim(left=-0.02, right=xlim)

    ax.set_ylabel("Probability", size=18)
    ax.set_xlabel("1st Waiting Latency (s/token)", size=18)

    ax.legend(fontsize=16)

    dataset_name = DATASETS_META[dataset]["name"]
    model_name = MODELS_META[model]["name"]
    ax.set_title(f"{dataset_name} | {model_name} | {req_rate}", size=20)

    filename = f"first_wait_latency_cdf_cdf_{dataset}_{model}_{req_rate}_{seed}_{duration}.pdf"
    output_path = os.path.join(saved_dir, filename)

    print(f"Saving {filename}...")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print("Finished.")

    # Plot Wait Latency
    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    for block_size, lats in zip(block_sizes, full_waiting_again_lat):
        x = np.sort(lats)
        y = 1. * np.arange(len(lats)) / (len(lats) - 1)

        ax.plot(x, y, color=BLOCK_SIZE_TO_COLOR[str(block_size)], label=block_size)

    if xlim:
        ax.set_xlim(left=-0.02, right=xlim)

    ax.set_ylabel("Probability", size=18)
    ax.set_xlabel("Waiting Latency (s/token)", size=18)

    ax.legend(fontsize=16)

    dataset_name = DATASETS_META[dataset]["name"]
    model_name = MODELS_META[model]["name"]
    ax.set_title(f"{dataset_name} | {model_name} | {req_rate}", size=20)

    filename = f"wait_latency_cdf_cdf_{dataset}_{model}_{req_rate}_{seed}_{duration}.pdf"
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
    parser.add_argument("--xlim", type=float, default=None)
    args = parser.parse_args()

    save_dir_abs_path = os.path.abspath(args.save_dir)
    block_sizes = [int(blk_size) for blk_size in args.block_sizes.split(",")]

    plot_latency_breakdown_cdf(save_dir_abs_path, args.dataset, args.model, block_sizes,
                        args.req_rate, args.seed, args.duration, args.xlim)
    
# python plot/latency_breakdown_cdf.py ./figures/latency_breakdown_cdf/ --dataset alpaca \
# --block-sizes 1,2,4,8,16,32,64,128,256 --model opt-13b --req-rate 31.5 \
# --xlim 0.5