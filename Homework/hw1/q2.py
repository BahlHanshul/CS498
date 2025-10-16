from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def server(params, optimizer, num_ranks):
    # ---- aggregate grads from workers ----
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous() #usage: tranfer a list tensor to one 1-D tensor
    ##here, you should generate one big 1-D tensor containing all parameters to make the transfer process easy
    grad_buf = flat_grad.clone() #agg as a aggregated counter to record sum gradients
    recv_buf = torch.empty_like(flat_grad)
    temp_val = 0
    src_rank = 1

    #                                                                   #
    #                                                                   #
    # your code here: receive gradients form worker, and add them to agg#
    #                                                                   #
    #                                                                   #
    while src_rank < num_ranks:
        temp_val += 1
        req = dist.irecv(recv_buf, src=src_rank)
        req.wait()
        grad_buf.add_(recv_buf)
        src_rank += 1

    grad_buf.div_(float(num_ranks))

    synced_grads = _unflatten_dense_tensors(grad_buf, [p.grad for p in params])
    # ---- set averaged grads locally & step ----
    for g, s in zip([p.grad for p in params], synced_grads):
        g.copy_(s)
    optimizer.step()

    # ---- broadcast updated params for this subset ----
    flat_params = _flatten_dense_tensors([p.data for p in params]).contiguous()
    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D parameter tensor to all workers   #
    #                                                                   #
    #                                                                   #
    for dst in range(1, num_ranks):
        
        send_req = dist.isend(flat_params, dst=dst)
        send_req.wait()


def worker(params):
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    # ---- push grads to server ----
    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D gradient to server
    #                                                                   #
    #                                                                   #
    send_handle = dist.isend(flat_grad, dst=0)
    send_handle.wait()

    # ---- receive updated params, write into local model ----
    #                                                                   #
    #                                                                   #
    # your code here: please get correct 1-D packed parameter from server
    #           And then unpacked it and store in synced_params
    #                                                                   #
    ref_param = _flatten_dense_tensors([p.data for p in params]).contiguous()
    recv_tensor = torch.empty_like(ref_param)
    recv_op = dist.irecv(recv_tensor, src=0)
    recv_op.wait()
    synced_params = _unflatten_dense_tensors(recv_tensor, [p.data for p in params]) #you should  assign correct value for synced_params#

    # ---- syncronize the parameters ----
    for p, s in zip(params, synced_params):
        p.data.copy_(s)


def PS_grads_(model, world_size=None, rankid=None, opt=None):
    """
    Synchronous PS step:
      - Rank 0: receive grads, sum/avg, set grads, opt.step(), broadcast updated params.
      - Rank >0: send grads, receive updated params, write into local model.
    Only processes the subset of parameters with non-None grads.
    """
    num_nodes = world_size
    rank = rankid

    # Fast path: single process
    single_process = (num_nodes == 1)
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    no_active_params = (len(params_with_grad) == 0)
    
    if single_process:
        opt.step()
        return

    #not necessary, as in most case, params won't be empty list...
    # Collect params that participated in this backward pass
    if no_active_params:
        # No grads this step; only server might still want to advance schedulers, etc.
        if rank == 0:
            opt.step()
        return
    b = 0
    temp_var = (rank == 0)
    if temp_var:
        server(params_with_grad, opt, num_nodes)
    else:
        worker(params_with_grad)

    # Optional hard step boundary
    dist.barrier()
