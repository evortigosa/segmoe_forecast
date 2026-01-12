# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Segment-wise Mixture-of-Experts (Seg-MoE)
Token-wise and Segment-wise MoE Modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .TransformerModel import FeedForward, ConvFeedForward, DwConvFeedForward



"""
Mixture-of-Experts (MoE)
"""


def get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias):
    if ffn_type == 'conv':
        return ConvFeedForward(d_model, d_ff, None, dropout, glu, bias)
    elif ffn_type == 'dwconv':
        return DwConvFeedForward(d_model, d_ff, None, dropout, glu, bias)
    else:  # ffn_type == 'mlp'
        return FeedForward(d_model, d_ff, None, dropout, glu, bias)


def get_expert_ffn(expert_type, d_model, d_ff, dropout, fan_gate, glu, bias):
    ffn_type= 'mlp' if expert_type == 'mlp' else 'mlp'

    return get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias)



class MoEFeedForward(nn.Module):
    """
    The Sparse Mixture-of-Experts (MoE) module. Delegate the modeling of diverse time series 
    patterns to sparse specialized experts in a data-driven manner through a sparce gating 
    function (only K of N experts per token) for expert assignments.
    - When n_experts=0, forward the input into a single FFN module; MoE otherwise.
    - ffn_type (str): defines the shared_expert type from 'mlp' for MLP-FFN, 'conv' for Conv-FFN,
    or 'dwconv' for DwConv-FFN.
    - experts_type (str): defines the routed experts from 'mlp' for MLP-FFN.
    See https://arxiv.org/abs/2410.10469 and https://arxiv.org/abs/2409.16040
    """

    def __init__(self, d_model, d_ff, dropout=0.2, ffn_type='mlp', fan_gate=False, glu=False,
                 n_experts=8, top_k=2, experts_type='mlp', bias=False) -> None:
        super(MoEFeedForward, self).__init__()
        assert n_experts >= 0, "n_experts must be non-negative"
        # store gating logits for auxiliary load-balancing regularizers (losses)
        self.router_logits= None

        # shared fallback expert -- ensures no token is unprocessed if its top-k experts happen
        # to be poorly trained or overflowed
        self.shared_expert= self.get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias)

        if n_experts == 0:
            self.experts= None
            self.top_k= 0
        else:
            assert top_k > 0, "top_k must be > 0"
            self.top_k= min(top_k, n_experts)

            if isinstance(experts_type, str):
                experts_type= [experts_type for _ in range(n_experts)]
            else:
                assert all(isinstance(item, str) for item in experts_type), \
                    "experts_type must be a list of strings"
                assert len(experts_type) >= n_experts, \
                    "experts_type must be a string or a list of length n_experts"

            # controls contribution from fallback expert
            self.shared_gating= nn.Linear(d_model, 1, bias=False)

            # n_experts routed expert modules
            self.experts= nn.ModuleList([
                self.get_expert_ffn(experts_type[i], d_model, d_ff, dropout, fan_gate, glu, bias)
                for i in range(n_experts)
            ])
            # experts gating to generate token-to-expert affinity scores
            self.gating= nn.Linear(d_model, n_experts, bias=False)

            # initialize gating modules with Glorot / fan_avg
            nn.init.xavier_uniform_(self.shared_gating.weight)
            nn.init.xavier_uniform_(self.gating.weight)


    def forward(self, x):
        B, T, C= x.size()

        # with no sparse routed experts
        if self.experts is None:
            return self.shared_expert(x)

        # with sparse routed experts
        x_squashed= x.view(-1, C)  # (B * T, C)

        # compute gating logits and probabilities via softmax
        self.router_logits= self.gating(x_squashed)  # (B * T, n_experts)
        router= F.softmax(self.router_logits.float(), dim=-1)
        # select top-k experts for each token (softmax scores and indices) -> (B * T, K)
        router, selected_experts= torch.topk(router, self.top_k, dim=-1)
        # renormalize over top-k so they sum to 1 -- keeps MoE as a convex mixture
        router= router / router.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # cast back to x dtype
        router= router.to(x.dtype)

        # one hot the selected experts -- (B * T, K, n_experts) -> (n_experts, K, B * T)
        expert_mask= F.one_hot(selected_experts, num_classes=len(self.experts)).permute(2, 1, 0)
        # output buffer
        results= torch.zeros_like(x_squashed)

        for expert_idx, expert in enumerate(self.experts):
            # expert_mask[i] tells us which (rank, token) pairs route to expert i
            # retrieve pairs where this expert is selected
            rank_idx, token_idx= torch.where(expert_mask[expert_idx])  # (K, B * T)
            # index the correct inputs and compute the expert output for the current expert
            # we route individual token embeddings, not whole sequences
            expert_inputs = x_squashed[None, token_idx].reshape(-1, C)
            routing_weight= router[token_idx, rank_idx, None]

            # apply expert and routing weight by gate
            current_expert= expert(expert_inputs) * routing_weight
            results.index_add_(0, token_idx, current_expert.to(x_squashed.dtype))

        # shared fallback expert always applied
        shared_out= self.shared_expert(x) * F.sigmoid(self.shared_gating(x))
        results= results.view(B, T, C) + shared_out

        return results.contiguous()



class MoESegmentV0(nn.Module):
    """
    The Sparse Mixture of Experts (MoE) module for token-segments. Delegate the modeling of diverse
    time series patterns to sparse specialized experts in a data-driven manner through a sparce
    gating function (only K of N experts per segment of tokens) for expert assignments.
    - When n_experts=0, forward the input into a single FFN module; MoE otherwise.
    - exp_segment_size: number of contiguous tokens contained in a token-segment (non-overlapping)
    to feed the routed experts (enable within-segment interactions learning). When exp_segment_size=1,
    each routed expert consumes individual token embeddings (standard MoE).
    """

    def __init__(self, d_model, d_ff, dropout=0.2, ffn_type='mlp', fan_gate=False, glu=False,
                 n_experts=4, top_k=1, experts_type='mlp', exp_segment_size=1, bias=False) -> None:
        super(MoESegmentV0, self).__init__()
        assert n_experts >= 0, "n_experts must be non-negative"
        top_k= top_k if n_experts > 0 else 0
        self.segment_size= int(exp_segment_size)
        self.temperature= 1.0
        # store gating logits for auxiliary load-balancing regularizers (losses)
        self.router_logits= None

        # shared fallback expert -- ensures no segment is unprocessed if its top-k experts happen
        # to be poorly trained or overflowed
        self.shared_expert= get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias)

        if n_experts == 0:
            self.experts= None
            self.top_k= 0
        else:
            assert top_k > 0, "top_k must be > 0"
            self.top_k= min(top_k, n_experts)
            assert self.segment_size > 0, "exp_segment_size must be > 0"

            if isinstance(experts_type, str):
                experts_type= [experts_type for _ in range(n_experts)]
            else:
                assert all(isinstance(item, str) for item in experts_type), \
                    "experts_type must be a list of strings"
                assert len(experts_type) >= n_experts, \
                    "experts_type must be a string or a list of length n_experts"

            # per-segment dims
            d_model_seg= d_model * self.segment_size
            self.in_proj= nn.Linear(d_model_seg, d_model, bias=bias) if self.segment_size > 1 else nn.Identity()

            # controls contribution from fallback expert
            self.shared_gating= nn.Linear(d_model, 1, bias=False)

            # n_experts routed expert modules -- if segment_size > 1, experts consume segments of
            # patches (token embeddings) to allow experts to learn within-segment interactions,
            # i.e., cross-token nonlinear combinations.
            self.experts= nn.ModuleList([
                get_expert_ffn(experts_type[i], d_model, d_ff, dropout, fan_gate, glu, bias)
                for i in range(n_experts)
            ])
            # experts gating to generate segment-pooled affinity scores
            self.gating= nn.Linear(d_model, n_experts, bias=False)

            self.out_proj= nn.Linear(d_model, d_model_seg, bias=bias) if self.segment_size > 1 else nn.Identity()

            # initialize Linear modules with Glorot / fan_avg
            for m in (self.in_proj, self.out_proj):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

            # initialize gating modules with Glorot / fan_avg
            nn.init.xavier_uniform_(self.shared_gating.weight)
            nn.init.xavier_uniform_(self.gating.weight)


    def forward(self, x):
        B, T, C= x.size()

        # no sparse routed experts
        if self.experts is None:
            return self.shared_expert(x)

        # form non-overlapping segments: the sequence is right-padded with zeros if needed and a
        # mask is used so padding does not contribute to outputs
        s= self.segment_size
        # pad sequence to multiple of s
        remainder= T % s
        if remainder > 0:
            pad_len   = s - remainder
            x_padded  = torch.cat([x, x.new_zeros((B, pad_len, C))], dim=1)  # (B, T + pad_len, C)
            valid_mask= torch.cat([
                x.new_ones((B, T), dtype=torch.bool), x.new_zeros((B, pad_len), dtype=torch.bool)
            ], dim=1)
        else:
            x_padded  = x
            valid_mask= x.new_ones((B, T), dtype=torch.bool)

        Tpad= x_padded.size(1)
        Segs= Tpad // s  # number of segments per batch element
        # reshape into segments: (B, Segs, s * C)
        x_padded= x_padded.contiguous().view(B, Segs, s * C)
        x_padded= self.in_proj(x_padded)  # (B, Segs, C)
        # flatten batch and segments to single dimension
        flat_segments= x_padded.view(-1, C)  # (B * Segs, C)

        # compute gating logits and probabilities via softmax on segment-pooled vectors
        self.router_logits= self.gating(flat_segments)  # (B * Segs, n_experts)
        router= F.softmax(self.router_logits.float() / max(1e-8, float(self.temperature)), dim=-1)
        # select top-k experts for each token (softmax scores and indices) -> (B * Segs, K)
        router, selected_experts= torch.topk(router, self.top_k, dim=-1)
        # renormalize over top-k so they sum to 1 -- keeps MoE as a convex mixture
        router= router / router.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # cast back to x dtype
        router= router.to(x.dtype)

        # output buffer
        results= torch.zeros_like(flat_segments)
        # one hot the selected experts -- (B * Segs, K, n_experts) -> (n_experts, K, B * Segs)
        expert_mask= F.one_hot(
            selected_experts, num_classes=len(self.experts)
        ).permute(2, 1, 0).to(device=x.device, dtype=torch.long)

        for expert_idx, expert in enumerate(self.experts):
            # expert_mask[i] tells us which (rank, token segment) pairs route to expert i
            # retrieve pairs where this expert is selected
            rank_idx, segment_idx= torch.where(expert_mask[expert_idx])  # (K, B * Segs)
            if segment_idx.numel() == 0:
                continue
            # index the correct inputs and compute the expert output for the current expert
            # gather token segment embeddings, not individual tokens or whole sequences
            expert_inputs = flat_segments.index_select(0, segment_idx)
            # get the corresponding gate weights
            routing_weight= (router[segment_idx, rank_idx].unsqueeze(-1)).to(expert_inputs.dtype)

            # apply expert and routing weight by gate and scatter-add to results
            current_expert= expert(expert_inputs) * routing_weight
            results.index_add_(0, segment_idx, current_expert.to(flat_segments.dtype))

        # reshape results_segments back into token sequence shape (B, Segs, C)
        results= results.contiguous().view(B, Segs, -1)

        # shared fallback expert always applied (to segmented inputs to avoid architectural asymmetry)
        shared_out= self.shared_expert(x_padded) * F.sigmoid(self.shared_gating(x_padded))

        results= self.out_proj(results + shared_out)
        results= results.contiguous().view(B, Tpad, -1)

        # ensure no contributions for padded tokens (mask-out)
        if remainder > 0:
            mask= valid_mask.unsqueeze(-1)  # (B, Tpad, 1)
            results= results * mask.to(results.dtype)
            results= results[:, :T, :]  # remove any padding

        return results



class MoESegment(nn.Module):
    """
    The Sparse Mixture of Experts (MoE) module for token-segments. Delegate the modeling of diverse
    time series patterns to sparse specialized experts in a data-driven manner through a sparce
    gating function (only K of N experts per segment of tokens) for expert assignments.
    - When n_experts=0, forward the input into a single FFN module; MoE otherwise.
    - exp_segment_size: number of contiguous tokens contained in a token-segment (non-overlapping)
    to feed the routed experts (enable within-segment interactions learning). When exp_segment_size=1,
    each routed expert consumes individual token embeddings (standard MoE).
    """

    def __init__(self, d_model, d_ff, dropout=0.2, ffn_type='mlp', fan_gate=False, glu=False,
                 n_experts=4, top_k=1, experts_type='mlp', exp_segment_size=1, bias=False) -> None:
        super(MoESegment, self).__init__()
        assert n_experts >= 0, "n_experts must be non-negative"
        top_k= top_k if n_experts > 0 else 0
        self.segment_size= int(exp_segment_size)
        self.temperature= 1.0
        # store gating logits for auxiliary load-balancing regularizers (losses)
        self.router_logits= None

        if n_experts == 0:
            self.shared_expert= get_ffn(ffn_type, d_model, d_ff, dropout, fan_gate, glu, bias)
            self.experts= None
            self.top_k= 0
        else:
            assert top_k > 0, "top_k must be > 0"
            self.top_k= min(top_k, n_experts)
            assert self.segment_size > 0, "exp_segment_size must be > 0"

            if isinstance(experts_type, str):
                experts_type= [experts_type for _ in range(n_experts)]
            else:
                assert all(isinstance(item, str) for item in experts_type), \
                    "experts_type must be a list of strings"
                assert len(experts_type) >= n_experts, \
                    "experts_type must be a string or a list of length n_experts"

            # per-segment dims
            d_model_seg= d_model * self.segment_size
            d_ff_seg= d_ff * self.segment_size

            # shared fallback expert -- ensures no segment is unprocessed if its top-k experts happen
            # to be poorly trained or overflowed
            self.shared_expert= get_ffn(ffn_type, d_model_seg, d_ff_seg, dropout, fan_gate, glu, bias)
            # controls contribution from fallback expert
            self.shared_gating= nn.Linear(d_model_seg, 1, bias=False)

            # n_experts routed expert modules -- if segment_size > 1, experts consume segments of
            # size d_model * segment_size to allow experts to learn within-segment interactions,
            # i.e., cross-token nonlinear combinations. This is not the same as applying a per-token
            # expert independently to each token of the segment
            self.experts= nn.ModuleList([
                nn.Sequential(
                    Rearrange('b (s c) -> b s c', s=self.segment_size) if self.segment_size > 1 else nn.Identity(),
                    get_expert_ffn(experts_type[i], d_model, d_ff, dropout, fan_gate, glu, bias),
                    Rearrange('b s c -> b (s c)') if self.segment_size > 1 else nn.Identity(),
                ) for i in range(n_experts)
            ])
            # experts gating to generate segment-pooled affinity scores
            self.gating= nn.Linear(d_model_seg, n_experts, bias=False)

            # initialize gating modules with Glorot / fan_avg
            nn.init.xavier_uniform_(self.shared_gating.weight)
            nn.init.xavier_uniform_(self.gating.weight)


    def forward(self, x):
        B, T, C= x.size()

        # no sparse routed experts
        if self.experts is None:
            return self.shared_expert(x)

        # form non-overlapping segments: the sequence is right-padded with zeros if needed and a
        # mask is used so padding does not contribute to outputs
        s= self.segment_size
        # pad sequence to multiple of s
        remainder= T % s
        if remainder > 0:
            pad_len   = s - remainder
            x_padded  = torch.cat([x, x.new_zeros((B, pad_len, C))], dim=1)  # (B, T + pad_len, C)
            valid_mask= torch.cat([
                x.new_ones((B, T), dtype=torch.bool), x.new_zeros((B, pad_len), dtype=torch.bool)
            ], dim=1)
        else:
            x_padded  = x
            valid_mask= x.new_ones((B, T), dtype=torch.bool)

        Tpad= x_padded.size(1)
        Segs= Tpad // s  # number of segments per batch element
        # reshape into segments: (B, Segs, s, C)
        x_padded= x_padded.contiguous().view(B, Segs, s, C)
        # flatten batch and segments to single dimension
        flat_segments= x_padded.view(-1, s * C)  # (B * Segs, s * C)

        # compute gating logits and probabilities via softmax on segment-pooled vectors
        self.router_logits= self.gating(flat_segments)  # (B * Segs, n_experts)
        router= F.softmax(self.router_logits.float() / max(1e-8, float(self.temperature)), dim=-1)
        # select top-k experts for each token (softmax scores and indices) -> (B * Segs, K)
        router, selected_experts= torch.topk(router, self.top_k, dim=-1)
        # renormalize over top-k so they sum to 1 -- keeps MoE as a convex mixture
        router= router / router.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # cast back to x dtype
        router= router.to(x.dtype)

        # output buffer
        results= torch.zeros_like(flat_segments)
        # one hot the selected experts -- (B * Segs, K, n_experts) -> (n_experts, K, B * Segs)
        expert_mask= F.one_hot(
            selected_experts, num_classes=len(self.experts)
        ).permute(2, 1, 0).to(device=x.device, dtype=torch.long)

        for expert_idx, expert in enumerate(self.experts):
            # expert_mask[i] tells us which (rank, token segment) pairs route to expert i
            # retrieve pairs where this expert is selected
            rank_idx, segment_idx= torch.where(expert_mask[expert_idx])  # (K, B * Segs)
            if segment_idx.numel() == 0:
                continue
            # index the correct inputs and compute the expert output for the current expert
            # gather token segment embeddings, not individual tokens or whole sequences
            expert_inputs = flat_segments.index_select(0, segment_idx)
            # get the corresponding gate weights
            routing_weight= (router[segment_idx, rank_idx].unsqueeze(-1)).to(expert_inputs.dtype)

            # apply expert and routing weight by gate and scatter-add to results
            current_expert= expert(expert_inputs) * routing_weight
            results.index_add_(0, segment_idx, current_expert.to(flat_segments.dtype))

        # reshape results_segments back into token sequence shape (B, Tpad, C)
        results= results.contiguous().view(B, Tpad, C)

        # shared fallback expert always applied (to segmented inputs to avoid architectural asymmetry)
        flat_segments= x_padded.view(B, Segs, -1)  # (B, Segs, s * C)
        shared_out= self.shared_expert(flat_segments) * F.sigmoid(self.shared_gating(flat_segments))
        shared_out= shared_out.contiguous().view(B, Tpad, C)

        results= results + shared_out

        # ensure no contributions for padded tokens (mask-out)
        if remainder > 0:
            mask= valid_mask.unsqueeze(-1)  # (B, Tpad, 1)
            results= results * mask.to(results.dtype)
            results= results[:, :T, :]  # remove any padding

        return results
