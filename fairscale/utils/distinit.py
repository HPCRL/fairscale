
import functools
import inspect
import logging
import multiprocessing
import os
import random
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from torch.distributed import rpc

def torch_version() -> Tuple[int, ...]:
    numbering = torch.__version__.split("+")[0].split(".")[:3]
    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    if not numbering[2].isnumeric():
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second usecase more than the first,
        # return the pre-release or dev numbering
        logging.warning(f"Pytorch pre-relase version {torch.__version__} - assuming intent to test it")
        numbering[2] = "0"
    return tuple(int(n) for n in numbering)


def dist_init(rank: int, world_size: int, filename: str, filename_rpc: str = "", backend="gloo") -> bool:
    """
    Initialize torch distributed, based on a temporary file shared across ranks, which makes it possible for unrelated
    tests to be run concurrently.
    .. warning: This limits the usecase to all ranks being on the same node
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    url = "file://" + filename
    url_rpc = "file://" + filename_rpc

    print(f"dist init r={rank}, world={world_size}")

    # CPU as backend
    if backend == "gloo":
        dist.init_process_group(backend=backend, init_method=url, world_size=world_size, rank=rank)
        return True
    else:
        # GPU as backend
        if torch_version() >= (1, 6, 0):
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if backend == "nccl" and torch.cuda.device_count() < world_size:
                logging.warning("Requested world size cannot be reached on this machine, not enough GPUs")
                return False

            dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=url)
            rpc.init_rpc(
                f"Test{rank}",
                rank=rank,
                world_size=world_size,
                backend=rpc.BackendType.TENSORPIPE,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=url_rpc),
            )
        else:
            if world_size > 1:
                # TensorPipe is not available in Torch 1.5
                rpc.init_rpc(
                    name=f"Test{rank}",
                    rank=rank,
                    world_size=world_size,
                    rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(init_method=url_rpc),
                )
            elif torch.cuda.is_available():
                dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method=url)
            else:
                return False

        if torch.cuda.is_available() and torch.cuda.device_count():
            torch.cuda.set_device(rank % torch.cuda.device_count())

    return True


