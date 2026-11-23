# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Asymmetric WY backward adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.common.backends.triton_ascend.chunk_delta_h import _pad_dense_tail_tokens, need_dma_mask, time_axis_needs_mask
from fla.ops.gated_delta_rule.backends.triton_ascend import wy_fast
from fla.ops.precond_gated_delta_rule.chunk_precond_wy_bwd import prepare_precond_wy_repr_bwd
from fla.ops.utils import prepare_chunk_indices
from fla.utils import ascend_compile_kwargs, input_guard

# Extra k_precond / dk_precond tiles vs the symmetric GDN finalize_k.
PRECOND_BWD_K_MEM_MULT = wy_fast.PREPARE_BWD_K_MEM_MULT + 1.0


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def prepare_precond_wy_repr_bwd_finalize_k_npu(
    k, k_precond, beta, dA_out, dk, dk_precond, db,
    cu_seqlens, chunk_indices, T, B,
    task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, KP: tl.constexpr,
    IS_VARLEN: tl.constexpr, BETA_T_CONTIG: tl.constexpr, DB_T_CONTIG: tl.constexpr,
):
    T_seq = T
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b, i_h = i_bh // HV, i_bh % HV
        i_t_o64 = tl.cast(i_t_o, tl.int64)
        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_t_o64 * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t_o64 * 2 + 1).to(tl.int64)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            i_t = i_t_o64
            bos = tl.cast(i_b, tl.int64) * T_seq

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if BETA_T_CONTIG:
            beta_ptr = wy_fast.get_gate_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            b_b = tl.load(beta_ptr + o_t, mask=m_t, other=0).to(tl.float32)
        else:
            b_b = tl.load(beta + (bos * HV + i_h) + o_t * HV, mask=m_t, other=0).to(tl.float32)
        if DB_T_CONTIG:
            db_ptr = wy_fast.get_gate_base(db, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            b_db = tl.load(db_ptr + o_t, mask=m_t, other=0).to(tl.float32)
        else:
            b_db = tl.load(db + (bos * HV + i_h) + o_t * HV, mask=m_t, other=0).to(tl.float32)
        o_i = tl.arange(0, BT)
        b_dA = tl.load(
            dA_out + (bos * HV + i_h) * BT + o_i[:, None] + o_t[None, :] * (HV * BT),
            mask=m_t[None, :], other=0,
        ).to(tl.float32)
        i_key = i_h // (HV // H)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_tk = m_t[:, None] & (o_k[None, :] < K)
            p_k = k + (bos * H + i_key) * K + o_t[:, None] * (H * K) + o_k[None, :]
            p_kp = k_precond + (bos * H + i_key) * K + o_t[:, None] * (H * K) + o_k[None, :]
            p_dk = dk + (bos * HV + i_h) * KP + o_t[:, None] * (HV * KP) + o_k[None, :]
            p_dkp = dk_precond + (bos * HV + i_h) * KP + o_t[:, None] * (HV * KP) + o_k[None, :]
            b_k = tl.load(p_k, mask=m_tk, other=0).to(tl.float32)
            b_kp = tl.load(p_kp, mask=m_tk, other=0).to(tl.float32)
            b_kb = b_k * b_b[:, None]
            # lhs dot consumes its copy; the original dA stays for the rhs dot.
            b_dA_lhs = b_dA + 0.0
            b_dkb = tl.dot(b_dA_lhs, b_kp, allow_tf32=False)
            b_db += tl.sum(b_dkb * b_k, 1)
            b_dk = b_dkb * b_b[:, None]
            b_dk += tl.load(p_dk, mask=m_tk, other=0).to(tl.float32)
            b_dkp = tl.trans(tl.dot(tl.trans(b_kb), b_dA, allow_tf32=False))
            tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_tk)
            tl.store(p_dkp, b_dkp.to(dk_precond.dtype.element_ty), mask=m_tk)

        if DB_T_CONTIG:
            tl.store(db_ptr + o_t, b_db.to(db.dtype.element_ty), mask=m_t)
        else:
            tl.store(db + (bos * HV + i_h) + o_t * HV, b_db.to(db.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=['T'])
def prepare_precond_wy_repr_bwd_finalize_a2_dg_npu(
    k, k_precond, beta, dA_out, dg,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, BETA_T_CONTIG: tl.constexpr, DG_T_CONTIG: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    """Fuse A_asymm = (beta * k) @ k_precond^T with dg += row(dA*A) - col(dA*A)."""
    i_t = tl.cast(tl.program_id(0), tl.int64) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if BETA_T_CONTIG:
        beta_ptr = wy_fast.get_gate_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        b_b = tl.load(beta_ptr + o_t, mask=m_t, other=0).to(tl.float32)
    else:
        b_b = tl.load(beta + (bos * HV + i_h) + o_t * HV, mask=m_t, other=0).to(tl.float32)
    if DG_T_CONTIG:
        dg_ptr = wy_fast.get_gate_base(dg, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
    o_i = tl.arange(0, BT)
    b_A_asymm = tl.zeros([BT, BT], dtype=tl.float32)
    i_key = i_h // (HV // H)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        b_k = tl.load(
            k + (bos * H + i_key) * K + o_t[:, None] * (H * K) + o_k[None, :],
            mask=m_tk, other=0,
        ).to(tl.float32)
        b_kp = tl.load(
            k_precond + (bos * H + i_key) * K + o_t[:, None] * (H * K) + o_k[None, :],
            mask=m_tk, other=0,
        ).to(tl.float32)
        b_kb = b_k * b_b[:, None]
        b_A_asymm = tl.dot(b_kb, tl.trans(b_kp), b_A_asymm, allow_tf32=False)
    b_dA = tl.load(
        dA_out + (bos * HV + i_h) * BT + o_i[:, None] + o_t[None, :] * (HV * BT),
        mask=m_t[None, :], other=0,
    ).to(tl.float32)
    b_prod = b_dA * b_A_asymm
    if DG_T_CONTIG:
        b_dg = tl.load(dg_ptr + o_t, mask=m_t, other=0).to(tl.float32)
    else:
        b_dg = tl.load(dg + (bos * HV + i_h) + o_t * HV, mask=m_t, other=0).to(tl.float32)
    b_dg += tl.sum(b_prod, axis=1) - tl.sum(b_prod, axis=0)
    if DG_T_CONTIG:
        tl.store(dg_ptr + o_t, b_dg.to(dg.dtype.element_ty), mask=m_t)
    else:
        tl.store(dg + (bos * HV + i_h) + o_t * HV, b_dg.to(dg.dtype.element_ty), mask=m_t)


def prepare_precond_wy_repr_bwd_verifier(k, k_precond, v, *args, **kwargs):
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    for tensor in (k, k_precond, v):
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "npu":
            return False, "input device is not NPU"
    return True, None


@register(prepare_precond_wy_repr_bwd, backend=TritonAscendBackend, verifier=prepare_precond_wy_repr_bwd_verifier)
@input_guard
def prepare_precond_wy_repr_bwd_npu(
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = A.shape[-1]
    is_ascend950 = torch.npu.get_device_name(k.device).startswith("Ascend950")
    if is_ascend950 and T:
        boundaries = [0, T] if cu_seqlens is None else cu_seqlens.tolist()
        padded_k, padded_v = triton.next_power_of_2(K), triton.next_power_of_2(V)
        if K != padded_k or V != padded_v or any((end - start) % BT for start, end in zip(boundaries, boundaries[1:])):
            # keep the masked time and feature lanes out of the WY backward Cube reductions on 950
            parts = [[] for _ in range(5)]
            for start, end in zip(boundaries, boundaries[1:]):
                length = end - start
                if not length:
                    continue
                padded_t = triton.cdiv(length, BT) * BT
                tensors = []
                for tensor, feature_pad in (
                    (k, padded_k - K), (k_precond, padded_k - K), (v, padded_v - V),
                    (beta, 0), (A, 0), (dw, padded_k - K), (du, padded_v - V),
                ):
                    segment = tensor[:, start:end].contiguous()
                    if feature_pad:
                        segment = torch.nn.functional.pad(segment, (0, feature_pad))
                    tensors.append(_pad_dense_tail_tokens(segment, padded_t))
                g_segment = None if g is None else _pad_dense_tail_tokens(g[:, start:end], padded_t, repeat_last=True)
                dk_seg, dkp_seg, dv_seg, db_seg, dg_seg = prepare_precond_wy_repr_bwd_npu(
                    *tensors[:4], g_segment, *tensors[4:],
                )
                for dest, result in zip(parts, (dk_seg[..., :K], dkp_seg[..., :K], dv_seg[..., :V], db_seg)):
                    dest.append(result[:, :length])
                if dg_seg is not None:
                    dg_valid = dg_seg[:, :length].contiguous()
                    if length != padded_t:
                        dg_valid[:, -1] += dg_seg[:, length:].sum(1)
                    parts[4].append(dg_valid)
            return tuple(torch.cat(group, dim=1) if group else None for group in parts)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = wy_fast.get_wy_block_size(BT, K, wy_fast.PREPARE_BWD_KV_MEM_MULT, wy_fast.MAX_TILE_BWD_KV)
    BV = wy_fast.get_wy_block_size(BT, V, wy_fast.PREPARE_BWD_KV_MEM_MULT, wy_fast.MAX_TILE_BWD_KV)
    BK_FIN = wy_fast.get_wy_block_size(BT, K, PRECOND_BWD_K_MEM_MULT, wy_fast.MAX_TILE_BWD)
    use_g = g is not None
    is_varlen = cu_seqlens is not None
    # Shared WY kernels mask partial chunks, so the unwritten tail stays zero-filled.
    zero_ws = time_axis_needs_mask(T, BT, cu_seqlens)
    KP = K
    dk = k.new_zeros(B, T, HV, K) if zero_ws else k.new_empty(B, T, HV, K)
    dk_precond = k.new_zeros(B, T, HV, K) if zero_ws else k.new_empty(B, T, HV, K)
    dv = v.new_zeros(B, T, HV, V) if zero_ws else v.new_empty(B, T, HV, V)
    db, db_t_contig = wy_fast.allocate_time_buffer(B, T, HV, dtype=beta.dtype, device=k.device)
    beta_arg, beta_t_contig = wy_fast.prepare_beta_input(beta, HV)
    dg, dg_t_contig = None, False
    g_gate, g_t_contig = None, False
    g_k_arg = k
    g_exp_precomp = False
    if use_g:
        dg, dg_t_contig = wy_fast.allocate_time_buffer(B, T, HV, dtype=g.dtype, device=k.device)
        g_gate, g_t_contig = wy_fast.prepare_gate_input(g, HV)
        g_k_arg = g_gate
        if not is_varlen:
            g_k_arg = torch.exp2(g_gate.float()).to(g_gate.dtype)
            g_exp_precomp = True
    dg_arg = dg if use_g else beta
    dA_scr = torch.empty_like(A, dtype=torch.float32)
    dA_mid = torch.empty_like(A, dtype=torch.float32)
    dA_out = torch.empty_like(A)

    mask_dma = need_dma_mask(T, BT, K, BK, V, BV, is_varlen, cu_seqlens)
    mask_time = time_axis_needs_mask(T, BT, cu_seqlens)
    base = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BT,
        IS_VARLEN=is_varlen,
    )
    core_base = dict(B=B, **base)
    task_num = NT * B * HV
    wy_fast.launch_wy_core_grid(
        wy_fast.prepare_wy_repr_bwd_kv_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            k=k,
            v=v,
            beta=beta_arg,
            g=g_k_arg,
            A=A,
            dw=dw,
            du=du,
            dk=dk,
            dv=dv,
            dA=dA_scr,
            db=db,
            dg=dg_arg,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_G=use_g,
            G_T_CONTIG=g_t_contig,
            BETA_T_CONTIG=beta_t_contig,
            DG_T_CONTIG=dg_t_contig,
            DB_T_CONTIG=db_t_contig,
            MASK_DMA=mask_dma,
            G_EXP_PRECOMP=g_exp_precomp,
            # Rounding the gated operands before dA biases the per-head ATK scale gradient.
            DA_FP32=is_ascend950 and k.dtype == torch.bfloat16,
            **core_base,
        ),
    )
    wy_fast.launch_wy_kernel(
        wy_fast.prepare_wy_repr_bwd_da_mask_dot1_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A,
            dA_scr=dA_scr,
            dA_mid=dA_mid,
            HV=HV,
            MASK_DMA=mask_time,
            **base,
        ),
    )
    wy_fast.launch_wy_kernel(
        wy_fast.prepare_wy_repr_bwd_da_dot2_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A,
            dA_mid=dA_mid,
            dA_out=dA_out,
            HV=HV,
            MASK_DMA=mask_time,
            **base,
        ),
    )
    if use_g:
        wy_fast.launch_wy_kernel(
            wy_fast.prepare_wy_repr_bwd_da_gate_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                g=g_gate,
                dA_out=dA_out,
                HV=HV,
                G_T_CONTIG=g_t_contig,
                MASK_DMA=mask_time,
                **base,
            ),
        )
    wy_fast.launch_wy_core_grid(
        prepare_precond_wy_repr_bwd_finalize_k_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            k=k,
            k_precond=k_precond,
            beta=beta_arg,
            dA_out=dA_out,
            dk=dk,
            dk_precond=dk_precond,
            db=db,
            H=H,
            HV=HV,
            K=K,
            BK=BK_FIN,
            KP=KP,
            BETA_T_CONTIG=beta_t_contig,
            DB_T_CONTIG=db_t_contig,
            **core_base,
        ),
    )
    if use_g:
        wy_fast.launch_wy_kernel(
            prepare_precond_wy_repr_bwd_finalize_a2_dg_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                k=k,
                k_precond=k_precond,
                beta=beta_arg,
                dA_out=dA_out,
                dg=dg_arg,
                H=H,
                HV=HV,
                K=K,
                BK=BK_FIN,
                BETA_T_CONTIG=beta_t_contig,
                DG_T_CONTIG=dg_t_contig,
                **base,
                **ascend_compile_kwargs(),
            ),
        )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
        dk_precond = dk_precond.view(B, T, H, HV // H, K).sum(3)
    if db_t_contig:
        db = db.transpose(1, 2).contiguous()
    if use_g and dg_t_contig:
        dg = dg.transpose(1, 2).contiguous()
    return dk, dk_precond, dv, db, dg
