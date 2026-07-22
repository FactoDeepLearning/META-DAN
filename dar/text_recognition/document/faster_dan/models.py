#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import relu
from torch.nn import Conv1d, Dropout
from torch.nn import Embedding
from torch.nn import Module
from dar.generic.models import GlobalAttDecoder, PositionalEncoding1DOnTheFly, PositionalEncoding2D


class FasterDANDecoder(Module):
    """
    Faster DAN decoder module
    """
    def __init__(self, params):
        super(FasterDANDecoder, self).__init__()
        self.params = params
        self.enc_dim = params["enc_dim"]
        self.dec_l_max = params["l_max"]

        self.dropout = Dropout(params["dec_dropout"])
        self.dec_att_win = params["attention_win"] if params["attention_win"] is not None else 1

        self.att_decoder = GlobalAttDecoder(params)

        self.emb = Embedding(num_embeddings=params["vocab_size"]+3, embedding_dim=self.enc_dim)

        self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim//2)
        self.pe_2d = PositionalEncoding2D(self.enc_dim, params["h_max"], params["w_max"])

        vocab_size = params["vocab_size"] + 1
        self.end_conv = Conv1d(self.enc_dim, vocab_size, kernel_size=1)

    def forward(self, features_1d, tokens, reduced_size, token_len, features_size, padding_value=None,
                cache=None, num_pred=None, keep_all_weights=False, line_indices=None, index_in_lines=None, two_step_parallel=False):
        device = features_1d.device
        # Token to Embedding
        emb_tokens = self.emb(tokens).permute(0, 2, 1)
        # Add 1D Positional Encoding

        pos_tokens = emb_tokens + torch.cat([self.pe_1d(line_indices), self.pe_1d(index_in_lines)], dim=1)

        pos_tokens = pos_tokens.permute(2, 0, 1)

        if num_pred is None:
            num_pred = tokens.size(1)

        num_tokens_to_keep = pos_tokens.size(0)
        if two_step_parallel and self.dec_att_win is not None:
            num_tokens_to_keep = min([num_tokens_to_keep, (num_pred+1)*self.dec_att_win])
            if self.dec_att_win is not None and cache is not None:
                cache = cache[:, -num_tokens_to_keep+num_pred:]
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        # Generate dynamic masks
        target_mask = self.generate_target_mask(tokens.size(1), device, line_indices, index_in_lines, two_step_parallel, num_pred)  # Use only already predicted tokens (causal)
        memory_mask = None  # Use all feature position

        # Generate static masks
        key_target_mask = self.generate_token_mask(tokens, token_len, device, padding_value)  # Use all token except padding
        key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)  # Use all feature position except padding

        target_mask = target_mask[..., -num_pred:, -num_tokens_to_keep:]
        key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, weights, cache = self.att_decoder(pos_tokens, memory_key=features_1d,
                                        memory_value=features_1d,
                                        tgt_mask=target_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=key_target_mask,
                                        memory_key_padding_mask=key_memory_mask,
                                        use_cache=True,
                                        cache=cache,
                                        predict_last_n_only=num_pred,
                                        keep_all_weights=keep_all_weights)

        dp_output = self.dropout(relu(output))
        preds = self.end_conv(dp_output.permute(1, 2, 0))

        if not keep_all_weights:
            weights = weights.reshape(weights.size(0), weights.size(1), features_size[2], features_size[3])
        return output, preds, cache, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        """
        Generate mask for encoded features
        """
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, tokens, token_len, device, padding_value):
        """
        Generate mask for tokens per sample
        """
        batch_size, len_max = tokens.size()
        mask = torch.zeros((batch_size, len_max), dtype=torch.bool, device=device)
        return mask

    def generate_target_mask(self, target_len, device, line_indices=None, index_in_lines=None, two_step_parallel=False, num_lines=None):
        """
        Generate mask for tokens per time step (teacher forcing)
        """

        if not two_step_parallel:
            batch_size = line_indices.size(0)
            mask = torch.ones((batch_size, target_len, target_len), dtype=torch.bool, device=device)
            index_first_pass, index_second_pass = line_indices == 0, line_indices != 0
            # first pass: triangle mask
            mask[index_first_pass] = torch.triu(mask, diagonal=1)[index_first_pass]
            # second pass:
            index_visible = index_in_lines.unsqueeze(2) >= index_in_lines.unsqueeze(1)
            index_visible = torch.logical_and(torch.logical_or(index_visible, line_indices.unsqueeze(1) ==0), torch.logical_not(torch.logical_and(line_indices.unsqueeze(1) ==0, index_in_lines.unsqueeze(1) == 0)))
            index_visible = torch.logical_not(index_visible)
            mask[index_second_pass] = index_visible[index_second_pass]
        else:
            batch_size = line_indices.size(0)
            mask = torch.zeros((batch_size, target_len, target_len), dtype=torch.bool, device=device)
            mask[:, :num_lines] = torch.tril(torch.ones((num_lines, target_len), dtype=torch.bool, device=device), diagonal=0)
            mask[:, num_lines:, :num_lines] = True
            mask[:, num_lines:, num_lines:] = (index_in_lines.unsqueeze(1) <= index_in_lines.unsqueeze(2))[:, num_lines:, num_lines:]
            mask = torch.logical_not(mask)
        mask = torch.repeat_interleave(mask.unsqueeze(1), dim=1, repeats=self.params['dec_num_heads'])  # (B, L, L) -> (B, H, L, L)
        b, h, l, _ = mask.size()
        mask = mask.reshape(b*h, l, l)  # (B, H, L, L) -> (B*H, L, L)
        return mask

    def get_pos_features(self, features):
        return self.pe_2d(features)