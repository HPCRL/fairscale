# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing ShardedDDP
"""

from contextlib import suppress
import copy
import tempfile
from typing import List

import numpy as np
import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.utils.testing import GPT2, check_same_model_params, skip_if_no_cuda, skip_if_py38, skip_if_single_gpu


def run_one_step(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    def check(broadcast_buffers: bool, grad_accumulation: bool = False) -> None:
        # Any model works. Add one different buffer per rank
        model = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))
        model.register_buffer("test_buffer", torch.ones((1)) * rank)
        model.to(device)

        next(model.parameters()).requires_grad = False  # Test non-trainable parameters

        optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
        ddp_model = ShardedDataParallel(model, optimizer, broadcast_buffers=broadcast_buffers)

        def check_same_model_params(same_params: bool):
            # Check that all the params are the same on all ranks
            # This should be true with and without broadcast_buffers, we don't have any real buffer here
            receptacle: List[torch.Tensor] = []

            if dist.get_backend() != "nccl":
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        # Check the params
                        receptacle = [p.clone() for _ in range(world_size)] if rank == 0 else []
                        dist.gather(p, receptacle, dst=0)
                        if rank == 0:
                            for sync_p in receptacle[1:]:
                                if same_params:
                                    assert torch.all(torch.eq(receptacle[0], sync_p)), "Models differ in between ranks"
                                else:
                                    assert not torch.all(
                                        torch.eq(receptacle[0], sync_p)
                                    ), "Gradients should not have been synced"

                # Check that all the buffers are in sync (authoritative rank is 0, its buffer is 0)
                if broadcast_buffers:
                    for b in ddp_model.buffers():
                        receptacle = [b.clone() for _ in range(world_size)] if rank == 0 else []
                        dist.gather(b, receptacle, dst=0)
                        if rank == 0:
                            for sync_b in receptacle[1:]:
                                if same_params:
                                    assert torch.all(torch.eq(receptacle[0], sync_b)), "Models differ in between ranks"
                                else:
                                    assert not torch.all(
                                        torch.eq(receptacle[0], sync_b)
                                    ), "Gradients should not have been synced"

                        assert b.cpu().item() == 0.0

        # The model should be synchronized in between the ranks at ShardedDataParallel construction time, check that
        check_same_model_params(same_params=True)

        # Optim loop
        def closure():
            optimizer.zero_grad()

            with ddp_model.no_sync() if grad_accumulation else suppress():
                input_tensor = torch.rand((64, 2)).to(device)
                loss = ddp_model(input_tensor).abs().sum()
                loss.backward()
            return loss

        # The models should stay the same in between the ranks
        for i in range(5):
            _ = optimizer.step(closure=closure)
            # when running on cpu/gloo the "nodes" are not really different
            same_params = device == torch.device("cpu") or grad_accumulation
            check_same_model_params(same_params=same_params)

    check(broadcast_buffers=False)
    check(broadcast_buffers=True)
    check(broadcast_buffers=False, grad_accumulation=True)
    check(broadcast_buffers=True, grad_accumulation=True)
    dist.destroy_process_group()


def run_test(backend, device, world_size=2):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_one_step, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


@skip_if_no_cuda
@skip_if_single_gpu
def test_step_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))


@skip_if_py38
def test_step_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"))


def run_ddp_parity(rank, world_size, backend, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    np.random.seed(rank)
    NUMBER_BATCHS = 5
    INPUTS = 2
    BATCH_SIZE = 32

    def check_parity(amp: bool, accumulate: bool, change_train_graph: bool):

        # The API should be the exact same in between the sharded and non-sharded variants, generic closure
        def closure(model, scaler, input_tensor, should_accumulate):
            accumulate_steps = 3 if should_accumulate else 1

            model.zero_grad()

            def step():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = model(input_tensor).abs().sum()
                        scaler.scale(loss).backward()
                else:
                    loss = model(input_tensor).abs().sum()
                    loss.backward()

            with model.no_sync() if should_accumulate else suppress():
                for _ in range(accumulate_steps - 1):
                    step()

            step()

        # Any model works. Add one different buffer per rank
        model = Sequential(Linear(INPUTS, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))
        model.register_buffer("test_buffer", torch.ones((1)) * rank)
        model.to(device)

        # Make sure that the model starts with non-trainable, so that we check for the buckets to be
        # properly reassigned when/if this changes
        next(model.parameters()).requires_grad = False

        sharded_optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-5, momentum=0.99)
        sharded_ddp_model = ShardedDataParallel(
            module=model, sharded_optimizer=sharded_optimizer, broadcast_buffers=True
        )

        ddp_model_single = copy.deepcopy(model)
        ddp_optimizer = torch.optim.SGD(ddp_model_single.parameters(), lr=1e-5, momentum=0.99)
        ddp_model = DDP(ddp_model_single, device_ids=[rank], broadcast_buffers=True, find_unused_parameters=True)

        ddp_scaler = TorchGradScaler() if amp else None
        sharded_ddp_scaler = ShardedGradScaler() if amp else None

        # The model should be synchronized in between the ranks at construction time, check that
        check_same_model_params(sharded_ddp_model, ddp_model)

        # Typical training loop, check that we get the exact same results as DDP
        for i in range(NUMBER_BATCHS):
            input_tensor = torch.rand((BATCH_SIZE, INPUTS)).to(device)

            def closure_ddp(input_tensor=input_tensor):
                return closure(ddp_model, ddp_scaler, input_tensor, accumulate)

            def closure_sharded(input_tensor=input_tensor):
                return closure(sharded_ddp_model, sharded_ddp_scaler, input_tensor, accumulate)

            # Step/scale both
            if ddp_scaler is not None:
                _ = closure_ddp(input_tensor)
                ddp_scaler.step(ddp_optimizer)
                ddp_scaler.update()
            else:
                ddp_optimizer.step(closure=closure_ddp)

            if sharded_ddp_scaler is not None:
                _ = closure_sharded(input_tensor)
                sharded_ddp_scaler.step(sharded_optimizer)
                sharded_ddp_scaler.update()
            else:
                sharded_optimizer.step(closure=closure_sharded)

            check_same_model_params(sharded_ddp_model, ddp_model, f"Rank: {rank} - Step {i} broke")

            # Flip the trainability of the first parameter back and forth
            if i == 0 and change_train_graph:
                next(sharded_ddp_model.parameters()).requires_grad = not next(
                    sharded_ddp_model.parameters()
                ).requires_grad
                next(ddp_model.parameters()).requires_grad = not next(ddp_model.parameters()).requires_grad
                check_same_model_params(sharded_ddp_model, ddp_model, f"Rank: {rank} - Trainability refresh {i} broke")

    # Test all combinations: AMP, Accumulate, Change train graph
    amp_tests = [False]
    if hasattr(torch.cuda.amp, "autocast"):
        amp_tests.append(True)

    for accumulate in [False, True]:
        for change_train_graph in [False, True]:
            for amp in amp_tests:
                print(
                    f"Checking configuration: accumulate {accumulate} - change train graph {change_train_graph} - amp {amp}"
                )
                check_parity(amp=amp, accumulate=accumulate, change_train_graph=change_train_graph)

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_ddp_parity():
    temp_file_name = tempfile.mkstemp()[1]
    world_size = torch.cuda.device_count()
    backend = dist.Backend.NCCL
    mp.spawn(run_ddp_parity, args=(world_size, backend, temp_file_name), nprocs=world_size, join=True)


def run_ddp_parity_two_optim(rank, world_size, backend, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    device = torch.device("cuda")
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    np.random.seed(rank)  # Any model works. Add one different buffer per rank

    model = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))
    model.register_buffer("test_buffer", torch.ones((1)) * rank)
    model.to(device)
    n_half_params = len(list(model.parameters())) // 2

    sharded_optimizer = OSS(
        params=list(model.parameters())[:n_half_params], optim=torch.optim.SGD, lr=1e-3, momentum=0.99
    )
    sharded_optimizer_2 = OSS(
        params=list(model.parameters())[n_half_params:], optim=torch.optim.SGD, lr=1e-3, momentum=0.99
    )

    sharded_ddp_model = ShardedDataParallel(module=model, sharded_optimizer=sharded_optimizer, broadcast_buffers=True)

    ddp_model_single = copy.deepcopy(model)
    ddp_optimizer = torch.optim.SGD(list(ddp_model_single.parameters())[:n_half_params], lr=1e-3, momentum=0.99)
    ddp_optimizer_2 = torch.optim.SGD(list(ddp_model_single.parameters())[n_half_params:], lr=1e-3, momentum=0.99)
    ddp_model = DDP(ddp_model_single, device_ids=[rank], broadcast_buffers=True)

    def check_same_model_params():
        for pg, ddp_pg in zip(sharded_optimizer.param_groups, ddp_optimizer.param_groups):
            for p, ddp_p in zip(pg["params"], ddp_pg["params"]):
                assert torch.allclose(
                    p, ddp_p, atol=1e-3
                ), f"Model parameters differ in between DDP and ShardedDDP {p} {ddp_p}"
        for b, ddp_b in zip(sharded_ddp_model.buffers(), ddp_model.buffers()):
            assert torch.allclose(b, ddp_b, atol=1e-3), "Model buffers differ in between DDP and ShardedDDP"

    check_same_model_params()  # The models should stay the same in between the ranks

    for i in range(20):
        input_tensor = torch.rand((64, 2)).to(device)

        # Run DDP
        ddp_optimizer.zero_grad()
        ddp_optimizer_2.zero_grad()
        ddp_loss = ddp_model(input_tensor).abs().sum()
        ddp_loss.backward()
        ddp_optimizer.step()
        ddp_optimizer_2.step()

        # Run Sharded
        sharded_optimizer.zero_grad()
        sharded_optimizer_2.zero_grad()
        sharded_loss = sharded_ddp_model(input_tensor).abs().sum()
        sharded_loss.backward()
        sharded_optimizer.step()
        sharded_optimizer_2.step()
        check_same_model_params()

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_ddp_parity_two_optim():
    temp_file_name = tempfile.mkstemp()[1]
    world_size = 2
    backend = dist.Backend.NCCL
    mp.spawn(run_ddp_parity_two_optim, args=(world_size, backend, temp_file_name), nprocs=world_size, join=True)


def run_test_two_inputs(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    class _DoubleInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))

        def forward(self, x, y):
            x1 = self.mlp(x)
            x2 = self.mlp(y)
            return torch.cat((x1, x2), dim=1)

    model = _DoubleInput().to(device)

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    # Optim loop
    def closure():
        optimizer.zero_grad()
        input_tensor = torch.rand((64, 2)).to(device)
        loss = ddp_model(input_tensor, input_tensor).abs().sum()
        loss.backward()
        return loss

    for i in range(5):
        _ = optimizer.step(closure=closure)

    dist.destroy_process_group()


def test_inputs():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cpu"
    mp.spawn(run_test_two_inputs, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def test_ddp_attributes():
    # Check that ShardedDDP exposes the same attributes as Pytorch's DDP
    # - is multi_device_module
    # - device_type

    url = "file://" + tempfile.mkstemp()[1]
    dist.init_process_group(init_method=url, backend="gloo", rank=0, world_size=1)

    model = Sequential(Linear(2, 3), Linear(3, 3))
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert hasattr(ddp_model, "is_multi_device_module")
    assert hasattr(ddp_model, "device_type")
    dist.destroy_process_group()


def test_random_attributes():
    # Check that ShardedDDP exposes the original module's attributes

    url = "file://" + tempfile.mkstemp()[1]
    dist.init_process_group(init_method=url, backend="gloo", rank=0, world_size=1)

    model = Sequential(Linear(2, 3), Linear(3, 3))
    model.banana = "sweet"

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert hasattr(ddp_model, "banana")
    assert not hasattr(ddp_model, "orange")

    dist.destroy_process_group()


def run_test_device_change(rank, world_size, backend, device, temp_file_name):
    # Check that the wrapped module can change devices

    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)

    model = Sequential(Linear(2, 3), Linear(3, 3)).cpu()
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)
    ddp_model.to(device)

    inputs = torch.rand((10, 2), device=device)
    outputs = ddp_model(inputs)  # assert if the module has not been changed properly
    loss = outputs.norm().backward()

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_device_change():
    # Check that ShardedDDP is compatible with sync batch norm across multiple GPUs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(run_test_device_change, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_test_ddp_sync_batch_norm(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)

    model = Sequential(Linear(2, 3), torch.nn.BatchNorm1d(3), Linear(3, 3)).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)  # in pytorch 1.5 syncBN switches to the default device/cpu

    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    assert isinstance(model[1], torch.nn.SyncBatchNorm)
    # Ensures sync batch norm handles have been added
    ddp_model(torch.randn(2, 2).to(device))
    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_ddp_sync_batch_norm():
    # Check that ShardedDDP is compatible with sync batch norm across multiple GPUs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(
        run_test_ddp_sync_batch_norm, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True
    )


def run_test_two_optimizers(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)

    class _DoubleInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Sequential(Linear(2, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3), Linear(3, 3))

        def forward(self, x, y):
            x1 = self.mlp(x)
            x2 = self.mlp(y)
            return torch.cat((x1, x2), dim=1)

    model = _DoubleInput().to(device)

    parameters = list(model.parameters())
    optimizer_1 = OSS(params=parameters[:-10], optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    optimizer_2 = OSS(params=parameters[-10:], optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, [optimizer_1, optimizer_2])

    # Optim loop
    def closure():
        input_tensor = torch.rand((64, 2)).to(device)
        loss = ddp_model(input_tensor, input_tensor).abs().sum()
        loss.backward()
        return loss

    for i in range(5):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        _ = optimizer_1.step(closure=closure)
        _ = optimizer_2.step(closure=closure)

    dist.destroy_process_group()


def test_two_optimizers():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cpu"
    mp.spawn(run_test_two_optimizers, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)


def run_test_gpt2(rank, world_size, backend, device, temp_file_name):
    INPUT_DIM = 32
    BACH_SIZE = 10
    STEPS = 10

    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    if device == torch.device("cuda"):
        torch.cuda.set_device(rank)

    torch.manual_seed(rank)
    np.random.seed(rank)
    model = GPT2(
        embed_dim=512, num_heads=2, num_layers=24, num_positions=INPUT_DIM * INPUT_DIM, num_vocab=512, num_classes=2
    ).to(device)
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=0.01, momentum=0.99)
    ddp_model = ShardedDataParallel(model, optimizer)

    # Optim loop
    def closure():
        optimizer.zero_grad()
        # Force int inputs to prevent the first grad from firing
        input_tensor = torch.randint(10, (BACH_SIZE, INPUT_DIM)).to(device)
        loss = ddp_model(input_tensor).abs().sum()
        loss.backward()
        return loss

    # Check for bucketing overflows
    for i in range(STEPS):
        _ = optimizer.step(closure=closure)

    dist.destroy_process_group()


@skip_if_no_cuda
@skip_if_single_gpu
def test_gpt2():
    # Check that the ShardedDDP wrapper accepts tuple(tensors) as inputs
    world_size = 2
    backend = "gloo"
    temp_file_name = tempfile.mkstemp()[1]
    device = "cuda"
    mp.spawn(run_test_gpt2, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)
