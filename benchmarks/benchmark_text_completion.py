import argparse
import logging
import os
import pickle
import time
from typing import List

from tqdm import tqdm

from trace import generate_text_completion_requests
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
    requests = generate_text_completion_requests(
        args.dataset,
        args.request_rate,
        args.duration,
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

    # Warm up.
    logger.info('Warming up.')
    num_warmup_requests = 8
    warmup_input_len = 8
    warmup_output_len = 32
    warmup_sampling_params = SamplingParams(
        n=1,
        best_of=1,
        temperature=1.0,
        top_p=0.99,
        use_beam_search=False,
        stop_token_ids=set(),
        max_tokens=warmup_output_len,
        logprobs=0,
    )

    llm.generate(
        sampling_params=warmup_sampling_params,
        prompt_token_ids=[[0] * warmup_input_len for _ in range(num_warmup_requests)]
    )

    # Start benchmarking.
    logger.info('Start benchmarking.')
    # Initialize tqdm.
    pbar = tqdm(total=len(requests), desc='Finished requests')

    outputs = []
    id_time_map = {}
    llm.llm_engine.scheduler.reset_stats()
    start_time = time.monotonic()
    while (requests or llm.llm_engine.has_unfinished_requests()):
        now = time.monotonic()
        if args.timeout is not None and now - start_time > args.timeout:
            logger.info('Timeout. Stop benchmarking.')
            break

        while requests:
            if requests[0][0] <= now - start_time:
                request_time, input_token_ids, sampling_params = requests.pop(0)
                request_id = str(next(llm.request_counter))
                llm.llm_engine.add_request(
                    request_id=request_id,
                    prompt=None,
                    sampling_params=sampling_params,
                    prompt_token_ids=input_token_ids,
                    arrival_time=start_time + request_time)
                id_time_map[request_id] = start_time + request_time
            else:
                break
        
        step_outputs = llm.llm_engine.step()

        now = time.monotonic()
        for output in step_outputs:
            if output.finished:
                arrival_time = id_time_map[output.request_id]
                finish_time = now
                outputs.append({
                    "output": output.to_dict(),
                    "arrival_time": arrival_time,
                    "finish_time": finish_time
                })
                pbar.update(1)

    pbar.close()

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
    parser.add_argument('--request-rate', type=float, help='reqs/sec', required=True)
    parser.add_argument('--duration', type=int, help='duration in seconds', required=True)
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
    parser.add_argument('--timeout', type=int, help='time out in seconds', default=None)
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
            './results',
            dataset_name,
            model_name,
            sample_dir,
            f'block{args.block_size}',
            f'req-rate-{args.request_rate}',
            f'seed{args.seed}',
            f'duration-{args.duration}',
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
