
import os
import tempfile

import pytest
import torch
from torch import nn
from torch.distributed import rpc
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel import layers
from fairscale.nn.pipe import Pipe
from fairscale.utils.testing import dist_init, get_world_sizes, set_random_seed, spawn_for_all_world_sizes, torch_spawn


def run_test_initialize_affine_weight(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing initialize_affine_weight with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    layers._initialize_affine_weight(weight, output_size, input_size, output_size_coeff, 0, torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    print ("orig master_weight size", master_weight.size()) 
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff, dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print(
        "   column parallel max error (should be zero) on global rank {}: {}".format(
            torch.distributed.get_rank(), error
        )
    )
    assert error < 1.0e-6


if __name__ == "__main__":
    # execute only if run as a script
    spawn_for_all_world_sizes(run_test_initialize_affine_weight)
