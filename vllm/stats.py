import os
import pickle
import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class SeqStats:
    physical: Dict[int, int] = field(default_factory=dict)

    logical: Dict[int, int] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return sum(num_tokens for num_tokens in self.physical.values())
    
    def add_physical_block(self, blk_num: int, num_tokens: int) -> None:
        self.physical[blk_num] = num_tokens

    def add_logical_block(self, blk_num: int, num_tokens: int) -> None:
        self.logical[blk_num] = num_tokens

    def to_dict(self):
        return {
            "physical": self.physical,
            "logical": self.logical,
            "seq_len": self.seq_len
        }

@dataclass
class SeqGroupStats:
    seqs: Dict[int, SeqStats] = field(default_factory=dict)

    def add_seq(self, id: int, stats: SeqStats) -> None:
        self.seqs[id] = stats

    def to_dict(self):
        out = {}

        for id, stats in self.seqs.items():
            out[id] = stats.to_dict()

        return out

@dataclass
class Analysis:
    seq_groups: Dict[str, SeqGroupStats] = field(default_factory=dict)

    def add_seq_group(self, id: int, stats: SeqGroupStats) -> None:
        self.seq_groups[id] = stats

    def to_dict(self):
        out = {}

        for id, stats in self.seq_groups.items():
            out[id] = stats.to_dict()

        return out

@dataclass
class Step:
    timestamp: float
    input_lens: int
    swap_out_lens: int
    swap_in_lens: int
    num_preempted: int
    num_to_running: int
    num_waiting: int
    num_running: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float

    num_logical_blocks: int
    num_logical_tokens: int
    num_physical_blocks: int
    num_physical_tokens: int

    analysis: Analysis

    def _analysis_to_dict(self):
        return {
            "timestamp": self.timestamp,
            "analysis": self.analysis.to_dict(),
        }
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "input_lens": self.input_lens,
            "swap_out_lens": self.swap_out_lens,
            "swap_in_lens": self.swap_in_lens,
            "num_preempted": self.num_preempted,
            "num_to_running":self.num_to_running,
            "num_waiting": self.num_waiting,
            "num_running": self.num_running,
            "num_swapped": self.num_swapped,
            "gpu_cache_usage": self.gpu_cache_usage,
            "cpu_cache_usage": self.cpu_cache_usage,
            "num_logical_blocks": self.num_logical_blocks,
            "num_logical_tokens": self.num_logical_tokens,
            "num_physical_blocks": self.num_physical_blocks,
            "num_physical_tokens": self.num_physical_tokens,
        }
    
@dataclass
class Stats:
    num_gpu_blocks: int
    num_cpu_blocks: int
    start_time: float = field(default=time.monotonic())
    steps: List[Step] = field(default_factory=list)
        
    def add_step(self, step: Step) -> None:
        self.steps.append(step)
    
    def reset(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.__init__(num_gpu_blocks, num_cpu_blocks)

    def _analysis_to_dict(self):
        out = []
        for step in self.steps:
            out.append(step._analysis_to_dict())
        
        return out

    def _stats_to_dict(self):
        steps = []
        for step in self.steps:
            steps.append(step.to_dict())
        
        out = {
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_cpu_blocks": self.num_cpu_blocks,
            "start_time": self.start_time,
            "steps": steps
        }

        return out
    
    def save(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, 'analysis.pkl'), 'wb') as f:
            pickle.dump(self._analysis_to_dict(), f)
        
        with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
            pickle.dump(self._stats_to_dict(), f)
