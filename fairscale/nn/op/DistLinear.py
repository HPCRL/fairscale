from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .utils import divide_and_check_no_remainder, initialize_affine_weight

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
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

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
            partition_strategy=[self.output_size_per_partition, self.reduce_size_per_partition]
        )

        #TODO:
        #1. how do we partition this kernel
        #2. Initialize weight? randomly ??
        #3. Input partition inside??
        #4. Do we need processor layout, I mean how to make it 2D or 3D??


    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # here we have big issue
            # how to distribute input if it has not been split
            # we need input_size_per_partition and reduce_size_per_partition
            input_parallel = scatter_to_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        if self.parti_reduce_features != 1:
            # TODO: need to all-reduce
            output_ = None

        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
