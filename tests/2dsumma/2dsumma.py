#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from datetime import timedelta

DEBUG = True

def summa(procGridX, procGridY, m, n, k, kk, A_block, B_block, C_block, C_ref_block):
    rank = dist.get_rank()

    print(f"[Rank {rank}] New call to summa function...\n")
    rowGroupIndex = [-1] * procGridX
    colGroupIndex = [-1] * procGridY

    indexX = rank % procGridX
    indexY = int(rank // procGridX)

    # generate row/col group
    for p in range(0, procGridX):
        rowGroupIndex[p] = indexY * procGridX + p
    for p in range(0, procGridY):
        colGroupIndex[p] = indexX + p * procGridX

    # if DEBUG:
    #     #print(f"[Rank {rank}] rowGroupIndex ", rowGroupIndex)
    #     #print(f"[Rank {rank}] colGroupIndex ", colGroupIndex)

    #######################
    # create all group map
    #######################
    row_group = None
    for i in range(0, procGridY):
        group_index = [-1] * procGridX
        for j in range(0, procGridX):
            group_index[j] = i * procGridX + j
        if i != indexY:
            dist.new_group(ranks=group_index, timeout=timedelta(seconds=25))
        else:
            row_group = dist.new_group(ranks=group_index, timeout=timedelta(seconds=25))

    col_group = None
    for i in range(0, procGridX):
        group_index = [-1] * procGridY
        for j in range(0, procGridY):
            group_index[j] = j * procGridX + i
        if i != indexX:
            dist.new_group(ranks=group_index, timeout=timedelta(seconds=25))
        else:
            col_group = dist.new_group(ranks=group_index, timeout=timedelta(seconds=25))

    buffer_A = torch.zeros(int(m // procGridY), kk)
    buffer_B = torch.zeros(kk, int(n // procGridX))
    buffer_C = torch.zeros(int(m // procGridY), int(n // procGridX))

    step = int(k // kk)
    index_x_cnt = 0
    index_y_cnt = 0
    local_index_x_cnt = 0
    local_index_y_cnt = 0
    # print(step)
    for s in range(0, step):
        panel_x_cnt = 0  # for computation buffer A, x dimension offset
        panel_y_cnt = 0  # for computation buffer B, y dimension offset

        aux_x_size = kk  # tile K for A
        aux_y_size = kk  # tile K for B

        n_ministep_per_kk_x = kk // (k // procGridX)  # processor x dimension -- A
        if kk % (k // procGridX) > 0:  # partial tile
            n_ministep_per_kk_x += 1

        n_ministep_per_kk_y = kk // (k // procGridY)  # processor y dimension -- B
        if kk % (k // procGridY) > 0:
            n_ministep_per_kk_y += 1

        whose_sending_on_row = index_x_cnt // (k // procGridX)  # which one is to bcast along row group
        whose_sending_on_col = index_y_cnt // (k // procGridY)  # which one is to bcast along col group
        # print(f"[Rank {rank}, step = {s}] whose_sending_on_row = {whose_sending_on_row}, whoseTurnCol = {whose_sending_on_col}, "
        #       f"nBlocks_X = {n_ministep_per_kk_x}, nBlocks_Y = {n_ministep_per_kk_y}")

        # send A
        while aux_x_size > 0:
            length_band = min(k // procGridX - local_index_x_cnt, aux_x_size)
            local_buffer_A = torch.zeros(m // procGridY, length_band)
            whose_sending_on_row = index_x_cnt // (k // procGridX)  # which one is to bcast along row group
            offset = indexY * procGridX

            if indexX == whose_sending_on_row:
                # fill local buffer A
                for i in range(0, m // procGridY):
                    for j in range(0, length_band):
                        local_buffer_A[i, j] = A_block[i, j + local_index_x_cnt]
                whose_sending_on_row += offset
                if DEBUG:
                    print(f"[Rank {rank}] indexX = {indexX} real_src = {whose_sending_on_row}"
                          f" broadcasting \n {local_buffer_A}")
                dist.broadcast(local_buffer_A, whose_sending_on_row, row_group)
            else:
                whose_sending_on_row += offset
                dist.broadcast(local_buffer_A, whose_sending_on_row, row_group)
            # fill bufferA
            for i in range(0, m // procGridY):
                for j in range(0, length_band):
                    buffer_A[i, j + panel_x_cnt] = local_buffer_A[i, j]

            index_x_cnt += length_band
            aux_x_size -= length_band

            local_index_x_cnt += length_band
            panel_x_cnt += length_band
            # resetting if move to next block but tile K still does not finish yet
            if local_index_x_cnt % (k // procGridX) == 0:
                local_index_x_cnt = 0

            if panel_x_cnt % kk == 0:
                panel_x_cnt = 0
        # end sending A

        # send B
        while aux_y_size > 0:
            length_band = min(k // procGridY - local_index_y_cnt, aux_y_size)
            local_buffer_B = torch.zeros(length_band, n // procGridX)
            whose_sending_on_col = index_y_cnt // (k // procGridY)  # which one is to bcast along col group
            offset = indexX

            if indexY == whose_sending_on_col:
                # fill local buffer B
                for i in range(0, length_band):
                    for j in range(0, n // procGridX):
                        local_buffer_B[i, j] = B_block[i + local_index_y_cnt, j]
                whose_sending_on_col = offset + whose_sending_on_col * procGridX
                if DEBUG:
                    print(f"[Rank {rank}] indexY = {indexY} real_src = {whose_sending_on_col}"
                          f" broadcasting \n {local_buffer_B}")
                dist.broadcast(local_buffer_B, whose_sending_on_col, col_group)
            else:
                whose_sending_on_col = offset + whose_sending_on_col * procGridX
                dist.broadcast(local_buffer_B, whose_sending_on_col, col_group)
            # fill bufferB
            for i in range(0, length_band):
                for j in range(0, n // procGridX):
                    buffer_B[i + panel_y_cnt, j] = local_buffer_B[i, j]

            index_y_cnt += length_band
            aux_y_size -= length_band

            local_index_y_cnt += length_band
            panel_y_cnt += length_band
            # resetting if move to next block but tile K still does not finish yet
            if local_index_y_cnt % (k // procGridY) == 0:
                local_index_y_cnt = 0

            if panel_y_cnt % kk == 0:
                panel_y_cnt = 0
        # end sending B

        # local matmal
        buffer_C += torch.matmul(buffer_A, buffer_B)
    # end steps loop
    if DEBUG:
        print(f" rank {rank} buffer_C {buffer_C}, size {buffer_C.size()}\n")
        print(f" rank {rank} C_ref_block {C_ref_block}, size {C_ref_block.size()}\n")

    # if rank == 1:
    #     print(torch.eq(C_ref_block, buffer_C))
    #     ref = torch.flatten(C_ref_block)
    #     c = torch.flatten(buffer_C)
    #     for i in range(0, len(c)):
    #         print(f"{i}, {c[i] - ref[i]}")
    # if torch.equal(C_ref_block, buffer_C):
    #     print("pass correctness")
    # else:
    #     print(f"wrong on processor rank {rank}")




def distribute_matrix(procGridX, procGridY, m, n, matrix, local_block, rank):
    chunk_matrix = []
    num_procs = procGridX * procGridY
    assert n % procGridX == 0, "evenly dispatch on processor space X"
    assert m % procGridY == 0, "evenly dispatch on processor space Y"
    block_y_size = m // procGridY
    block_x_size = n // procGridX
    if rank == 0:
        print("Split Matrix")
        for pid in range(0, num_procs):
            row = pid // procGridX
            col = pid % procGridX
            start_pos_row = row * block_y_size
            start_pos_col = col * block_x_size
            lblock = torch.zeros(block_y_size, block_x_size)
            for i in range(0, block_y_size):
                for j in range(0, block_x_size):
                    lblock[i, j] = matrix[start_pos_row + i, start_pos_col + j]
            # if DEBUG: print("local_block", lblock)
            chunk_matrix.append(lblock)
        # if DEBUG: print("size of chunk_matrix ", len(chunk_matrix))
        # if DEBUG: print("chunk_matrix \n", chunk_matrix)
    dist.scatter(local_block, chunk_matrix, src=0)


def init_process(rank, px, py, m, n, k, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    size = px * py
    dist.init_process_group(backend, rank=rank, world_size=size)

    tensor_A = None
    tensor_B = None
    tensor_C = None
    tensor_C_ref = None

    A_block = torch.zeros(m // py, k // px)
    B_block = torch.zeros(k // py, n // px)
    C_block = torch.zeros(m // py, n // px)
    C_ref_block = torch.zeros(m // py, n // px)

    if dist.get_rank() == 0:
        print("Random Matrix A, B")
        tensor_A = torch.randn(m, k)
        tensor_B = torch.randn(k, n)
        tensor_C = torch.zeros(m, n)
        print(f"tensor_A \n{tensor_A}")
        print(f"tensor_B \n{tensor_B}")
        print("local Reference")
        tensor_C_ref = torch.matmul(tensor_A, tensor_B)
        print(f"reference C \n size : {tensor_C_ref.size()} \n : {tensor_C_ref}")

    # initial distribute Matrix_A and Matrix_B
    distribute_matrix(px, py, m, k, tensor_A, A_block, rank)
    distribute_matrix(px, py, k, n, tensor_B, B_block, rank)
    distribute_matrix(px, py, m, n, tensor_C, C_block, rank)
    distribute_matrix(px, py, m, n, tensor_C_ref, C_ref_block, rank)

    # print(f"[Rank {dist.get_rank()}] show Matrix A block...\n{A_block}\n")
    # print(f"[Rank {dist.get_rank()}] show Matrix B block...\n{B_block}\n")
    kk = 2
    fn(px, py, m, n, k, kk, A_block, B_block, C_block, C_ref_block)


if __name__ == "__main__":
    px = 2
    py = 2
    m = 4
    n = 8
    k = 4

    num_procs = px * py
    size = num_procs

    # equivalent
    # processes = []
    # for rank in range(size):
    #     p = Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()

    mp.spawn(init_process, args=(px, py, m, n, k, summa), nprocs=size, join=True)
