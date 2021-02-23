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

from typing import Tuple

import torch

def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


# def initialize_affine_weight(
#     weight: torch.Tensor,
#     out_features: int,
#     reduce_size: int,
#     init_method: Callable[[torch.Tensor], torch.Tensor],
#     stride: int = 1,
#     return_master_weight: bool = False,
#     partition_size_list: List[int] = [1, 1],  #out_dim, reduction_dim, Here is transposed
# ) -> Optional[torch.Tensor]:
#     """Initialize affine weight for distribution."""
#     # If we only use 1 process for model parallelism, bypass scatter.
#     world_size = get_model_parallel_world_size()
#     if world_size == 1:
#         init_method(weight)
#         if return_master_weight:
#             return weight
#         return None
#
#     # Initialize master weight
#     master_weight = torch.empty(out_features, reduce_size, dtype=weight.dtype, requires_grad=False)
#     init_method(master_weight)
#
#     #partition large master weight into small piece (partition_size_list[0] x partition_size_list[1])
#     weight_list = torch.unfold(0, partition_size_list[0], partition_size_list[0])\
#         .unfold(1, partition_size_list[1], partition_size_list[1])
#
#     # TODO: rank how to map 2D processor space??
#     rank = get_model_parallel_rank()
#     local_weight_list = weight_list[??]
#
#     with torch.no_grad():
#         torch.cat(my_weight_list, dim=partition_dim, out=weight)
#     if return_master_weight:
#         return master_weight
#     return None
