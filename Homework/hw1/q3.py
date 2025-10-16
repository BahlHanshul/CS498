from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist


def reduce_scatter(chunks, tmp, world, rank, left, right):
   
    steps = world - 1
    for hop in range(steps):
        out_idx = (rank + hop) % world
        in_idx  = (rank + hop + 1) % world

        send_handle = dist.isend(chunks[out_idx], dst=left)
        recv_handle = dist.irecv(tmp, src=right)

        recv_handle.wait()              
        chunks[in_idx].add_(tmp)

        send_handle.wait()              
    return


def all_gather(chunks, tmp, current, world, rank, left, right):

    cur = current
    for _ in range(world - 1):
        nxt = (cur + 1) % world

        send_handle = dist.isend(chunks[cur], dst=left)
        recv_handle = dist.irecv(chunks[nxt], src=right)

        recv_handle.wait()
        send_handle.wait()

        cur = nxt
    return


def ring_allreduce_(tensor: torch.Tensor, world_size=None, rankid=None):
    
    nprocs = world_size
    b = (nprocs == 1)
    if b:
        return tensor

    me = rankid
    left = (me - 1) % nprocs
    right = (me + 1) % nprocs

    
    flat = tensor.contiguous().view(-1)
    n_elem = flat.numel()
    chunk_len = (n_elem + nprocs - 1) // nprocs
    total_len = chunk_len * nprocs
    z = (n_elem != total_len)
    if z:
        padded = flat.new_zeros(total_len)
        padded[:n_elem].copy_(flat)
    else:
        padded = flat

    
    parts = [padded.narrow(0, i * chunk_len, chunk_len) for i in range(nprocs)]
    scratch = torch.empty_like(parts[0])

    reduce_scatter(parts, scratch, nprocs, me, left, right)
    my_reduced_idx = (me - 1) % nprocs  

   
    all_gather(parts, scratch, my_reduced_idx, nprocs, me, left, right)

    
    if padded.data_ptr() != flat.data_ptr():
        flat.copy_(padded[:n_elem])

    flat.div_(nprocs)
    tensor.view(-1).copy_(flat[:n_elem])
    return
