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
from typing import Tuple, Callable, List, Optional

import torch
from fairscale.nn.mix import get_model_parallel_world_size, get_model_parallel_rank


def initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    reduce_size: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    stride: int = 1,
    return_master_weight: bool = False,
    partition_size_list: List[int] = [1, 1],  #out_dim, reduction_dim, Here is transposed
) -> Optional[torch.Tensor]:
    """Initialize affine weight for distribution."""
    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.range(out_features*reduce_size, dtype=torch.int32, requires_grad=False).reshape(out_features,reduce_size)

    #partition large master weight into small piece (partition_size_list[0] x partition_size_list[1])
    weight_list = torch.unfold(0, partition_size_list[0], partition_size_list[0])\
        .unfold(1, partition_size_list[1], partition_size_list[1])

    # TODO: rank how to map 2D processor space??
    # answer: need a p_space.
    # TODO: replica for case 4 ??

    rank = get_model_parallel_rank()
    local_weight_list = weight_list[rank]
    print("local_weight_list is {}".format(local_weight_list))

    with torch.no_grad():
        torch.cat(local_weight_list, out=weight)
        print("out is {}".format(weight))

    if return_master_weight:
        return master_weight
    return None
