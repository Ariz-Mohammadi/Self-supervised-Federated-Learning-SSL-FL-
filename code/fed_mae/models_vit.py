# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        # Map custom boolean global_pool to timm's string parameter
        if global_pool:
            kwargs['global_pool'] = 'avg'
            kwargs['class_token'] = False  # Disable class token for avg pooling
        else:
            kwargs['global_pool'] = 'token'  # Default timm behavior with class token
            kwargs['class_token'] = True

        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, C]
    
        # If using class token (token pooling)
        if getattr(self, "num_prefix_tokens", 0) > 0:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
            x = torch.cat((cls_tokens, x), dim=1)
    
        # Add positional embedding (always matches x.shape[1])
        x = x + self.pos_embed
        x = self.pos_drop(x)
    
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
    
        # DO NOT pool or normalize here. Just return tokens.
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model