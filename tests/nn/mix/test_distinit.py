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


import tempfile
import torch
import torch.multiprocessing as mp
from fairscale.utils.distinit import dist_init
from fairscale.nn.mix import initialize as mpu

def run_test_initialize_model_parallel(rank, model_parallel_size, filename, filename_rpc, backend):
    dist_init(rank, model_parallel_size, filename, filename_rpc, backend)

    if torch.distributed.get_rank() == 0:
        print("> testing initialize_model_parallel with size {} ...".format(model_parallel_size))
    model_parallel_size_ = min(model_parallel_size, torch.distributed.get_world_size())
    assert not mpu.model_parallel_is_initialized()
    mpu.initialize_model_parallel(model_parallel_size_)
    assert mpu.model_parallel_is_initialized()

    # Checks.
    def check(group, world_size, rank):
        assert world_size == torch.distributed.get_world_size(group=group)
        assert rank == torch.distributed.get_rank(group=group)

    # Model parallel.
    world_size = model_parallel_size_
    rank = torch.distributed.get_rank() % model_parallel_size_
    assert world_size == mpu.get_model_parallel_world_size()
    assert rank == mpu.get_model_parallel_rank()
    check(mpu.get_model_parallel_group(), world_size, rank)

    # Data parallel.
    world_size = torch.distributed.get_world_size() // model_parallel_size_
    rank = torch.distributed.get_rank() // model_parallel_size
    assert world_size == mpu.get_data_parallel_world_size()
    assert rank == mpu.get_data_parallel_rank()
    check(mpu.get_data_parallel_group(), world_size, rank)

    # Reset groups
    mpu.destroy_model_parallel()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(">> passed the test :-)")

def test_cpu_init():
    WORLD_SIZE = 4
    _, filename = tempfile.mkstemp()
    _, filename_rpc = tempfile.mkstemp()
    mp.spawn(run_test_initialize_model_parallel, args=(WORLD_SIZE, filename, filename_rpc, "gloo"), nprocs=WORLD_SIZE, join=True)


# def test_gpu_init():
#     WORLD_SIZE = 4
#     _, filename = tempfile.mkstemp()
#     _, filename_rpc = tempfile.mkstemp()
#     mp.spawn(run_test_initialize_model_parallel, args=(WORLD_SIZE, filename, filename_rpc, "nccl"), nprocs=WORLD_SIZE, join=True)
