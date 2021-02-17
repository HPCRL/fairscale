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
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel import layers
from fairscale.nn.pipe import Pipe
from fairscale.utils.testing import dist_init, get_world_sizes, set_random_seed, spawn_for_all_world_sizes, torch_spawn
import torch.multiprocessing as mp

class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight

def run_test_column_parallel_linear(rank, model_parallel_size, filename, filename_rpc):
    dist_init(rank, model_parallel_size, filename, filename_rpc)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing ColumnParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = layers.ColumnParallelLinear(input_size, output_size, keep_master_weight_for_test=True, gather_output=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    temp = torch.mul(output, loss_weight)
    loss = temp.sum()
    if torch.distributed.get_rank() == 0:
        #print ("input ", input_)
        print ("input dim ", input_.size())
        print ("output dim ", output.size())
        print ("loss weight dim", loss_weight.size())
        print ("temp dim", temp.size())
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    if torch.distributed.get_rank() == 0:
        print ("A dim ", A.size())
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdA on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    my_dLdb = torch.split(dLdb, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdb on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdX on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")



WORLD_SIZE=2
def test_row_parallel():
    _, filename = tempfile.mkstemp()
    _, filename_rpc = tempfile.mkstemp()
    args: Any = []
    # (lefaudeux) Let mp handle the process joining, join=False and handling context has been unstable in the past
    mp.spawn(run_test_column_parallel_linear, args=(WORLD_SIZE, filename, filename_rpc, *args), nprocs=WORLD_SIZE, join=True)
    #spawn_for_all_world_sizes(run_test_row_parallel_linear)

if __name__ == "__main__":
    test_row_parallel()
