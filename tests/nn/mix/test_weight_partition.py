# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import pytest
import torch
from torch import nn
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.mix import initialize as mpu
from fairscale.nn.mix import DistLinear
from fairscale.utils.distinit import dist_init


def run_test_weight_full_distribute(rank, model_parallel_size, filename, filename_rpc, backend):
    dist_init(rank, model_parallel_size, filename, filename_rpc, backend)
    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> model parallel size: {}".format(model_parallel_size))

    input_size = 4
    reduce_size = 2
    output_size = 4

    linear_layer = DistLinear(input_size, reduce_size, output_size, keep_master_weight_for_test=True,
                              partition_strategy=[2, 2, 2])
    # Reset groups
    mpu.destroy_model_parallel()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")

def test_weight_full_distribute():
    WORLD_SIZE = 4
    _, filename = tempfile.mkstemp()
    _, filename_rpc = tempfile.mkstemp()
    mp.spawn(run_test_weight_full_distribute, args=(WORLD_SIZE, filename, filename_rpc, "gloo"), nprocs=WORLD_SIZE,
             join=True)


