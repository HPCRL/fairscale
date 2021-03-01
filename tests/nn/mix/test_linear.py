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

class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.arange(m*n, dtype=torch.float32).reshape(m, n))

    def forward(self):
        return self.weight


def run_test_single_linear(rank, model_parallel_size, filename, filename_rpc, backend):
    dist_init(rank, model_parallel_size, filename, filename_rpc, backend)
    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    input_size = 4
    reduce_size = 2
    output_size = 4

    # Network
    identity_layer = IdentityLayer2D(input_size, reduce_size)
    linear_layer = DistLinear(input_size, reduce_size, output_size, keep_master_weight_for_test=True,
                              partition_strategy=[2, 2, 2])
    loss_weight = torch.ones([input_size, output_size], dtype=torch.float32)

    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight)

    if rank == 0:
        print(input_.size())
        print(input_)
        print(output.size())
        print(output)
        print(loss.size())
        print(loss)

    # Forward validate
    if rank == 0:
        t_a = torch.tensor(torch.arange(input_size*reduce_size, dtype=torch.float32)
                                      .reshape(input_size, reduce_size))

        t_b = torch.tensor(torch.arange(output_size*reduce_size, dtype=torch.float32)
                                      .reshape(output_size, reduce_size))

        t_c = torch.matmul(t_a, t_b.t())

        loss_ref = torch.mul(t_c, loss_weight)

        print(t_a.size())
        print(t_a)
        print(t_b.size())
        print(t_b)
        print(t_c.size())
        print(t_c)
        print(loss_ref.size())
        print(loss_ref)


        #assert(input_ == t_a)





    # # Backward
    # loss.sum().backward()
    # # Validate

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")

def test_single_linear_op():
    WORLD_SIZE = 4
    _, filename = tempfile.mkstemp()
    _, filename_rpc = tempfile.mkstemp()
    mp.spawn(run_test_single_linear, args=(WORLD_SIZE, filename, filename_rpc, "gloo"), nprocs=WORLD_SIZE,
             join=True)


