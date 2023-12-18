import argparse
import multiprocessing
import numpy as np
import os
import pickle

from utils import RESULTS_DIR

def process_results(
    dataset, model, block_size, req_rate, seed, duration
):
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

    analysis_path = os.path.join(input_dir, "analysis.pkl")
    with open(analysis_path, "rb") as f:
        analysis = pickle.load(f)

    # Process Data
    ts_adjusted = []
    for step in stats["steps"]:
        ts_adjusted.append(step["timestamp"]+stats["start_time"])

    req_ids = set()
    for step in analysis:
        for req_id in step["analysis"].keys():
            req_ids.add(int(req_id))

    finish_times = np.zeros(shape=(max(req_ids)+1, 1))

    for req in requests:
        finish_times[int(req["output"]["request_id"])] = req["finish_time"]

    arrival_times = np.zeros(shape=(max(req_ids)+1, 1))

    for req in requests:
        arrival_times[int(req["output"]["request_id"])] = req["arrival_time"]

    # Create Status Matrix
    # -1 = hasn't arrived or has finished
    # 0 = waiting
    # 1 = running

    status_matrix = np.empty(shape=(max(req_ids)+1,len(ts_adjusted)))
    status_matrix.fill(-1)

    for i, step in enumerate(analysis):
        for req_id in step["analysis"].keys():
            status_matrix[int(req_id)][i] = 1

    for i, row in enumerate(status_matrix):
        started = False

        arrival_time = arrival_times[i]
        finish_time = finish_times[i]
        
        for j, col in enumerate(row):
            current_time = ts_adjusted[j]

            # Capture waiting after starting
            if started == True and col == -1.0:
                if current_time < finish_time:
                    status_matrix[i][j] = 0.0
                else:
                    break
            
            # Capture waiting before starting
            elif started == False and col == -1.0:
                if current_time > arrival_time:
                    status_matrix[i][j] = 0.0
            
            # Capture 1st transition from -1 or 0 to 1
            elif started == False and col == 1.0:
                started = True

            # Continue without making a change
            else:
                continue

    # Create Tokens Matrix
    tokens_matrix = np.zeros(shape=(max(req_ids)+1,len(ts_adjusted)))

    for i, row in enumerate(status_matrix):
        started = False

        for j, col in enumerate(row):
            if col == 1.0:
                started = True
                tokens_matrix[i][j] = sum(x for x in analysis[j]["analysis"][str(i)][i]["logical"].values())
            if started == True and col == 0.0:
                tokens_matrix[i][j] = tokens_matrix[i][j-1]
            if started == True and col == -1.0:
                break

    # Save matrices
    status_matrix_path = os.path.join(input_dir, "status_matrix.pkl")

    print(f"Saving {status_matrix_path}...")
    with open(status_matrix_path, "wb") as handle:
        pickle.dump(status_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished.")

    tokens_matrix_path = os.path.join(input_dir, "tokens_matrix.pkl")

    print(f"Saving {tokens_matrix_path}...")
    with open(tokens_matrix_path, "wb") as handle:
        pickle.dump(tokens_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished.")

def process_results_helper(
    dataset, model, block_sizes, req_rate, seed, duration
):

    # dataset, model, block_size, req_rate, seed, duration
    inputs = []

    for block_size in block_sizes:
        inputs.append((dataset, model, block_size, req_rate, seed, duration))

    pool_obj = multiprocessing.Pool()
    pool_obj.starmap(process_results, inputs)
    pool_obj.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--block-sizes", type=str, required=True)
    parser.add_argument("--req-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=600)
    args = parser.parse_args()

    block_sizes = [int(blk_size) for blk_size in args.block_sizes.split(",")]

    process_results_helper(args.dataset, args.model, block_sizes, args.req_rate,
                           args.seed, args.duration)

# python plot/process_results.py --dataset alpaca --model opt-13b \
# --block-sizes 1,2,4,8,16,32,64,128,256 --req-rate 31.5