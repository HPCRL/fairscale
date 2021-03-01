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
#
# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


from typing import Tuple, Callable, List, Optional

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from . import get_model_parallel_world_size, get_model_parallel_group
from .utils import divide_and_check_no_remainder
from .weight_init import initialize_affine_weight

class DistLinear(torch.nn.Module):
    """
    Y = X*A + b
    X: input
    A: kernel
    Arguments:
        in_features: first dimension of matrix X (input dim)
        reduce_size: first dimension of matrix A (reduction dim).
        out_features: second dimension of matrix A (output dim).
        bias: If true, add bias. Note that bias is not parallelized.
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        partition_strategy: split in reduction dim and out dim
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        reduce_size: int,
        bias: bool = True,
        gather_output: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        partition_strategy: List[int] = [1, 1, 1],  #in_dim, out_dim, reduction_dim
        distributor_factor: List[int] = [1, 1, 1],  # in_dim, out_dim, reduction_dim  (1 means copy, n means /n split)
    ) -> None:
        super(DistLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.reduce_size = reduce_size
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel

        world_size = get_model_parallel_world_size() # need to define

        self.parti_in_features = partition_strategy[0]
        self.parti_out_features = partition_strategy[1]
        self.parti_reduce_features = partition_strategy[2]

        self.distributor_in_features = distributor_factor[0]
        self.distributor_out_features = distributor_factor[1]
        self.distributor_reduce_features = distributor_factor[2]

        # how to assert ??
        # assert (self.parti_in_features*self.parti_out_features) \
        #        <= world_size, "product of partition can not exceed all nodes number"

        self.input_size_per_partition = divide_and_check_no_remainder(in_features, self.parti_in_features)
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, self.parti_out_features)
        self.reduce_size_per_partition = divide_and_check_no_remainder(reduce_size, self.parti_reduce_features)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.reduce_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = initialize_affine_weight(
            self.weight,
            self.out_features,
            self.reduce_size,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
            partition_size_list=[self.output_size_per_partition, self.reduce_size_per_partition],
            processor_map=[partition_strategy[1], partition_strategy[2]]
        )

        #TODO for weight initial:
        #1. add distributor factor


    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore

        # TODO:here we have big issue 1 :: remapping ??
        # 1. how to get data if not in current node???
        # 2. Who to take responsibility for remapping?? (parent node sending?) (self pulling? )
        # 3. minor. how to distribute input if it has not been split \
        # we need input_size_per_partition and reduce_size_per_partition

        if self.input_is_parallel:
            input_parallel = input_
        else:
            group = get_model_parallel_group()
            rank = torch.distributed.get_rank(group=group)
            dim0_size = self.input_size_per_partition
            dim1_size = self.reduce_size_per_partition
            # partition large master weight into small piece (partition_size_list[0] x partition_size_list[1])
            input_list = input_.unfold(0, dim0_size, dim0_size).unfold(1, dim1_size, dim1_size)
            input_parallel = input_list[(rank % self.parti_reduce_features), (rank // self.parti_reduce_features), :, :]

        if self.parti_reduce_features * self.distributor_reduce_features != 1: #reduction
            for k in range(0, self.parti_reduce_features * self.distributor_reduce_features):
                output_parallel = F.linear(input_parallel, self.weight)
        else:
            output_parallel = F.linear(input_parallel, self.weight)

        if self.parti_reduce_features != 1:
            # TODO:
            # 1. need to all-reduce or all gather
            output_ = None

        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
