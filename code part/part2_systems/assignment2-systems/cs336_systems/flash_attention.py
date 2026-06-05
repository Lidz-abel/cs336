from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _attention_scores(q: Tensor, k: Tensor, is_causal: bool) -> Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_positions = torch.arange(n_queries, device=q.device)[:, None]
        k_positions = torch.arange(n_keys, device=q.device)[None, :]
        mask = q_positions >= k_positions
        scores = torch.where(mask, scores, torch.full_like(scores, -1e6))
    return scores


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        scores = _attention_scores(q, k, bool(is_causal))
        lse = torch.logsumexp(scores, dim=-1)
        p = torch.exp(scores - lse.unsqueeze(-1))
        out = torch.matmul(p, v)

        ctx.save_for_backward(q, k, v, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        q, k, v, lse = ctx.saved_tensors
        scores = _attention_scores(q, k, ctx.is_causal)
        p = torch.exp(scores - lse.unsqueeze(-1))

        grad_v = torch.matmul(p.transpose(-2, -1), grad_out)
        grad_p = torch.matmul(grad_out, v.transpose(-2, -1))
        grad_scores = p * (grad_p - torch.sum(grad_p * p, dim=-1, keepdim=True))

        scale = 1.0 / math.sqrt(q.shape[-1])
        grad_q = torch.matmul(grad_scores, k) * scale
        grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * scale
        return grad_q, grad_k, grad_v, None


if triton is not None and tl is not None:

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        lse_ptr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        d: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lq: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        batch = tl.program_id(0)
        query = tl.program_id(1)
        key_offsets = tl.arange(0, block_n)
        dim_offsets = tl.arange(0, block_d)
        key_mask = key_offsets < n_keys
        dim_mask = dim_offsets < d

        q = tl.load(q_ptr + batch * stride_qb + query * stride_qq + dim_offsets * stride_qd, mask=dim_mask, other=0.0)
        k = tl.load(
            k_ptr + batch * stride_kb + key_offsets[:, None] * stride_kk + dim_offsets[None, :] * stride_kd,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        scores = tl.sum(k * q[None, :], axis=1) * scale
        valid = key_mask
        if is_causal:
            valid = valid & (key_offsets <= query)
        scores = tl.where(valid, scores, -1.0e6)

        row_max = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - row_max)
        exp_sum = tl.sum(exp_scores, axis=0)
        lse = tl.log(exp_sum) + row_max
        p = exp_scores / exp_sum

        v = tl.load(
            v_ptr + batch * stride_vb + key_offsets[:, None] * stride_vk + dim_offsets[None, :] * stride_vd,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        out = tl.sum(p[:, None] * v, axis=0)
        tl.store(out_ptr + batch * stride_ob + query * stride_oq + dim_offsets * stride_od, out, mask=dim_mask)
        tl.store(lse_ptr + batch * stride_lb + query * stride_lq, lse)

    @triton.jit
    def _flash_backward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_out_ptr,
        lse_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        d: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_gob: tl.constexpr,
        stride_goq: tl.constexpr,
        stride_god: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lq: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        batch = tl.program_id(0)
        query = tl.program_id(1)
        key_offsets = tl.arange(0, block_n)
        dim_offsets = tl.arange(0, block_d)
        key_mask = key_offsets < n_keys
        dim_mask = dim_offsets < d

        q = tl.load(q_ptr + batch * stride_qb + query * stride_qq + dim_offsets * stride_qd, mask=dim_mask, other=0.0)
        do = tl.load(
            grad_out_ptr + batch * stride_gob + query * stride_goq + dim_offsets * stride_god,
            mask=dim_mask,
            other=0.0,
        )
        k = tl.load(
            k_ptr + batch * stride_kb + key_offsets[:, None] * stride_kk + dim_offsets[None, :] * stride_kd,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_ptr + batch * stride_vb + key_offsets[:, None] * stride_vk + dim_offsets[None, :] * stride_vd,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        scores = tl.sum(k * q[None, :], axis=1) * scale
        valid = key_mask
        if is_causal:
            valid = valid & (key_offsets <= query)
        scores = tl.where(valid, scores, -1.0e6)

        lse = tl.load(lse_ptr + batch * stride_lb + query * stride_lq)
        p = tl.exp(scores - lse)
        p = tl.where(valid, p, 0.0)
        dp = tl.sum(v * do[None, :], axis=1)
        delta = tl.sum(p * dp, axis=0)
        ds = p * (dp - delta)

        dq = tl.sum(ds[:, None] * k, axis=0) * scale
        tl.store(grad_q_ptr + batch * stride_qb + query * stride_qq + dim_offsets * stride_qd, dq, mask=dim_mask)

        dk = ds[:, None] * q[None, :] * scale
        dv = p[:, None] * do[None, :]
        tl.atomic_add(
            grad_k_ptr + batch * stride_kb + key_offsets[:, None] * stride_kk + dim_offsets[None, :] * stride_kd,
            dk,
            sem="relaxed",
            mask=valid[:, None] & dim_mask[None, :],
        )
        tl.atomic_add(
            grad_v_ptr + batch * stride_vb + key_offsets[:, None] * stride_vk + dim_offsets[None, :] * stride_vd,
            dv,
            sem="relaxed",
            mask=valid[:, None] & dim_mask[None, :],
        )


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        if triton is None or tl is None or not q.is_cuda:
            return FlashAttentionPytorch.apply(q, k, v, is_causal)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=q.dtype)
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        d = q.shape[-1]
        block_n = triton.next_power_of_2(n_keys)
        block_d = triton.next_power_of_2(d)

        _flash_forward_kernel[(q.shape[0], n_queries)](
            q,
            k,
            v,
            out,
            lse,
            n_queries,
            n_keys,
            d,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            1.0 / math.sqrt(d),
            bool(is_causal),
            block_n,
            block_d,
        )
        ctx.save_for_backward(q, k, v, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        q, k, v, lse = ctx.saved_tensors
        if triton is None or tl is None or not grad_out.is_cuda:
            scores = _attention_scores(q, k, ctx.is_causal)
            p = torch.exp(scores - lse.unsqueeze(-1))
            grad_v = torch.matmul(p.transpose(-2, -1), grad_out)
            grad_p = torch.matmul(grad_out, v.transpose(-2, -1))
            grad_scores = p * (grad_p - torch.sum(grad_p * p, dim=-1, keepdim=True))
            scale = 1.0 / math.sqrt(q.shape[-1])
            grad_q = torch.matmul(grad_scores, k) * scale
            grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * scale
            return grad_q, grad_k, grad_v, None

        grad_out = grad_out.contiguous()
        grad_q = torch.empty_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        d = q.shape[-1]
        block_n = triton.next_power_of_2(n_keys)
        block_d = triton.next_power_of_2(d)

        _flash_backward_kernel[(q.shape[0], n_queries)](
            q,
            k,
            v,
            grad_out,
            lse,
            grad_q,
            grad_k,
            grad_v,
            n_queries,
            n_keys,
            d,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            lse.stride(0),
            lse.stride(1),
            1.0 / math.sqrt(d),
            ctx.is_causal,
            block_n,
            block_d,
        )
        return grad_q, grad_k, grad_v, None
