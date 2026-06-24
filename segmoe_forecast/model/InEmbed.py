# -*- coding: utf-8 -*-
"""
# The Time-Series Forecasting Transformer (TSFT) with Segment-wise Mixture-of-Experts (Seg-MoE)
"""

import math
import torch
import torch.nn as nn
from .Normalization import RMSNorm



"""
# Input Modules
"""


class PositionalEmbedding(nn.Module):
    """
    Implements the standard PE function as in https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, block_size, d_model, base_val=10000.0) -> None:
        super(PositionalEmbedding, self).__init__()
        self.block_size= block_size
        self.base_val= base_val

        # create a long tensor of block_size positions
        position= torch.arange(0, block_size).unsqueeze(1)
        frequencies= torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(base_val) / d_model)
        )
        # create an empty placeholder
        wpe= torch.zeros(block_size, d_model).float()
        wpe.require_grad= False
        # iterating over each element in the sequence using sin and cos
        wpe[:, 0::2]= torch.sin(position * frequencies)
        wpe[:, 1::2]= torch.cos(position * frequencies)
        # register_buffer -- it is not saved in the state_dict nor optimized
        self.register_buffer('wpe', wpe.unsqueeze(0), persistent=False)


    def extra_repr(self):
        return f"block_size={self.block_size}, base={self.base_val}"


    def forward(self, x):
        x= x + self.wpe[:, : x.size(1)].to(x.device)

        return x



class PatchMasking(nn.Module):
    """
    Applies random masking to the patch embeddings for self-supervised pretraining tasks.
    A specified fraction (mask_ratio) of patches is set to zero. PatchMasking masks patches by zeroing
    them in place.
    being dropped, so the sequence length is unchanged.
    - If has_cls_tk=True, the class token (first token) is not masked.
    """

    def __init__(self, mask_ratio=0.75) -> None:
        super(PatchMasking, self).__init__()
        assert 0.0 <= mask_ratio < 1.0, "mask_ratio must be in [0, 1)"
        self.mask_ratio= mask_ratio


    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}"


    def forward(self, x):
        """ The masking mechanism is used only during self-supervised pretraining. """
        # ---- fast path: nothing to mask (fine-tuning / inference) ----
        if self.mask_ratio == 0.:
            return x

        # ---- general path: fixed-count random patch masking ----
        cls= None
        if self.has_cls_tk:
            # ensure the class token will not be masked
            cls= x[:, :1, :]
            x  = x[:, 1:, :]

        B, P, C= x.size()  # (batch_size, num_patches, d_model)
        # the masked subset is chosen uniformly at random per sample via a random shuffle of patch indices
        pto_keep= int(P * (1 - self.mask_ratio))
        ids_shuffle= torch.rand(B, P, dtype=x.dtype, device=x.device).argsort(dim=1)
        ids_restore= torch.argsort(ids_shuffle, dim=1)
        # binary mask in shuffled order (1=mask, 0=keep): first pto_keep are kept, then unshuffle
        mask= torch.ones(B, P, dtype=x.dtype, device=x.device)
        mask[:, :pto_keep]= 0
        mask= torch.gather(mask, dim=1, index=ids_restore).bool()  # (B, P): exactly P-pto_keep True/row
        # expand mask to match x dimensions (B, P, 1)
        mask= mask.unsqueeze(-1)

        # set masked positions to zero
        x= x.masked_fill(mask, value=0.0)
        if cls is not None:
            x= torch.cat((cls, x), dim=1)

        return x



class PatchEmbeddingV3(nn.Module):
    """
    Initializes the Embedding module. Applies either depthwise separable or regular convolutions
    to patch the input sequence and applies normalization + dropout.
    - v3: uses RMSNorm as the normalization layer.
    """

    def __init__(self, patch_width, channels, d_model, dropout=0.2, ch_independence=True) -> None:
        super(PatchEmbeddingV3, self).__init__()
        self.patch_width= patch_width
        self.d_model= d_model
        self.channels= 1 if ch_independence else channels

        # define convolutional patch embedding
        self.embed= nn.Conv1d(  # (batch_size, d_model, num_patches)
            self.channels, d_model, kernel_size=patch_width, stride=patch_width, bias=False
        )
        # define normalization and dropout modules for regularization
        self.norm= RMSNorm(d_model)
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None

        # initialize Conv modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


    def extra_repr(self):
        return f"patch_width={self.patch_width}, d_model={self.d_model}"


    def forward(self, ts):
        # ts -> (batch_size, channels/features, seq_length)
        B, C, T= ts.size()
        if self.channels == 1:
            ts= ts.reshape(-1, T).unsqueeze(1)  # (batch_size * channels/features, 1, seq_length)
            # ensure channel independence, batch_size assume batch_size * channels/features

        x= self.embed(ts)
        # x -> (B * C, d_model, num_patches)
        x= x.permute(0, 2, 1)
        # x -> (B * C, num_patches, d_model)
        if self.dropout is not None:
            x= self.dropout(self.norm(x))
        else:
            x= self.norm(x)

        return x.contiguous()  # (B * C, num_patches, d_model)



class PatchEmbedding(nn.Module):
    """
    Initializes the Embedding module. Applies either depthwise separable or regular convolutions
    to patch the input sequence and applies normalization + dropout.
    - v4: uses GroupNorm as the normalization layer.
    """

    def __init__(self, patch_width, channels, d_model, dropout=0.2, ch_independence=True) -> None:
        super(PatchEmbedding, self).__init__()
        self.patch_width= patch_width
        self.d_model= d_model
        self.channels= 1 if ch_independence else channels

        # define convolutional patch embedding
        self.embed= nn.Conv1d(  # (batch_size, d_model, num_patches)
            self.channels, d_model, kernel_size=patch_width, stride=patch_width, bias=False
        )
        # define normalization and dropout modules for regularization
        self.norm= nn.GroupNorm(num_groups=1, num_channels=d_model)
        # single group, equivalent with a LayerNorm
        self.dropout= nn.Dropout(p=dropout) if dropout > 0.0 else None

        # initialize Conv modules with Glorot / fan_avg
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def extra_repr(self):
        return f"patch_width={self.patch_width}, d_model={self.d_model}"


    def forward(self, ts):
        # ts -> (batch_size, channels/features, seq_length)
        B, C, T= ts.size()
        if self.channels == 1:
            ts= ts.reshape(-1, T).unsqueeze(1)  # (batch_size * channels/features, 1, seq_length)
            # ensure channel independence, batch_size assume batch_size * channels/features

        x= self.embed(ts)
        # x -> (B * C, d_model, num_patches)
        if self.dropout is not None:
            x= self.dropout(self.norm(x))
        else:
            x= self.norm(x)
        x= x.permute(0, 2, 1)
        # x -> (B * C, num_patches, d_model)

        return x.contiguous()  # (B * C, num_patches, d_model)
