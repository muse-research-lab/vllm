import argparse
import logging
import os
import pickle
import time
from typing import List

from tqdm import tqdm

from trace import generate_throughput_requests
from utils import get_model_name, get_dataset_name, get_dataset, get_sampling_dir_name

from vllm import LLM, SamplingParams


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    llm = LLM(
        model=args.model,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        block_size=args.block_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_sequences,
        max_model_len=args.max_model_len,
        disable_log_stats=not args.log_stats,
        collect_stats=args.collect_stats,
        token=args.token,
    )

    # Generate requests.
    requests = generate_throughput_requests(
        args.dataset,
        args.num_requests,
        args.seed,
        args.n1,
        args.n2,
        args.n3,
        args.n4,
        args.n6,
        args.n2_beam,
        args.n4_beam,
        args.n6_beam,
        args.n8_beam,
        args.max_model_len,
    )

    # Start benchmarking.
    logger.info('Start benchmarking.')

    # Add the requests to the engine.
    for input_token_ids, sampling_params in requests:
        request_id = str(next(llm.request_counter))
        llm.llm_engine.add_request(
            request_id=request_id,
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=input_token_ids)

    # Initialize tqdm.
    pbar = tqdm(total=len(requests), desc='Finished requests')

    outputs = []
    start_time = time.perf_counter()
    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()

        now = time.perf_counter()
        for output in step_outputs:
            if output.finished:
                finish_time = now
                outputs.append({
                    "output": output.to_dict(),
                    "arrival_time": start_time,
                    "finish_time": finish_time
                })
                pbar.update(1)

    # end time = finish time of the last step outputs
    end_time = now
    pbar.close()

    elapsed_time = end_time - start_time
    total_num_tokens = 0
    for output in outputs:
        prompt_len = len(output["output"]["prompt_token_ids"])

        output_len = 0
        for completion_output in output["output"]["outputs"].values():
            output_len += len(completion_output["token_ids"])

        total_num_tokens += prompt_len + output_len

    logger.info(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")

    logger.info('Finish benchmarking. Saving stats.')
    llm.llm_engine.scheduler.save_stats(args.output_dir)
    with open(os.path.join(args.output_dir, 'requests.pkl'), 'wb') as f:
        pickle.dump(outputs, f)
    
    logger.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark real text completion workloads.')
    
    # LLM parameters
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-memory-utilization', type=float, required=False)
    parser.add_argument('--swap-space', type=int, required=False)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--max-num-batched-tokens', type=int, required=False)
    parser.add_argument('--max-num-sequences', type=int, required=False)
    parser.add_argument('--max-model-len', type=int, required=False)
    parser.add_argument('--log-stats', action='store_true')

    # Requests parameters
    parser.add_argument('--dataset', type=str, help='path to dataset', required=True)
    parser.add_argument('--num-requests',type=int, default=1000, help='mumber of prompts to requests.')
    parser.add_argument('--n1', type=float, help='ratio of requests with n=1', default=0.0)
    parser.add_argument('--n2', type=float, help='ratio of requests with n=2', default=0.0)
    parser.add_argument('--n3', type=float, help='ratio of requests with n=3', default=0.0)
    parser.add_argument('--n4', type=float, help='ratio of requests with n=4', default=0.0)
    parser.add_argument('--n6', type=float, help='ratio of requests with n=6', default=0.0)
    parser.add_argument('--n2-beam', type=float, help='ratio of requests with n=2 & beam search', default=0.0)
    parser.add_argument('--n4-beam', type=float, help='ratio of requests with n=4 & beam search', default=0.0)
    parser.add_argument('--n6-beam', type=float, help='ratio of requests with n=6 & beam search', default=0.0)
    parser.add_argument('--n8-beam', type=float, help='ratio of requests with n=8 & beam search', default=0.0)

    # Other parameters
    parser.add_argument('--output-dir', type=str, help='path to output directory', default=None)
    parser.add_argument('--collect-stats', action='store_true')
    parser.add_argument('--token', type=str, help='Hugging Face token', default=None)

    args = parser.parse_args()
    
    if args.n1 + args.n2 + args.n3 + args.n4 + args.n6 + args.n2_beam + args.n4_beam + args.n6_beam + args.n8_beam != 1.0:
        raise ValueError('The ratios of requests must sum to 1.')

    model_name = get_model_name(args.model)
    dataset_name = get_dataset_name(args.dataset)
    
    args.dataset = get_dataset(dataset_name, model_name)
    
    sample_dir = get_sampling_dir_name(
        args.n1, args.n2, args.n3, args.n4, args.n6, args.n2_beam, args.n4_beam, args.n6_beam, args.n8_beam)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            './results-offline',
            dataset_name,
            model_name,
            sample_dir,
            f'block{args.block_size}',
            f'num-requests-{args.num_requests}',
            f'seed{args.seed}',
        )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'log.txt'), mode="w"),
        ],
    )
    logger.info(args)

    main(args)
