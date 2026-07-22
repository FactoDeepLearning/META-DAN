import torch
from torch import relu
from torch.nn import Conv1d, Dropout
from torch.nn import Embedding
from torch.nn import Module
from dar.generic.models import PositionalEncoding2D, GlobalAttDecoder, PositionalEncoding1D


class DANDecoder(Module):
    """
    DAN decoder module
    """
    def __init__(self, params):
        super().__init__()
        self.enc_dim = params["enc_dim"]
        self.dec_l_max = params["l_max"]

        self.dropout = Dropout(params["dec_dropout"])
        self.dec_att_win = params["attention_win"] if params["attention_win"] is not None else 1

        self.att_decoder = GlobalAttDecoder(params)

        self.emb = Embedding(num_embeddings=params["vocab_size"]+3, embedding_dim=self.enc_dim)
        self.pe_1d = PositionalEncoding1D(self.enc_dim, self.dec_l_max)
        self.pe_2d = PositionalEncoding2D(self.enc_dim, params["h_max"], params["w_max"])

        vocab_size = params["vocab_size"] + 1
        self.end_conv = Conv1d(self.enc_dim, vocab_size, kernel_size=1)

    def forward(self, features_1d, tokens, reduced_size, token_len, features_size, start=0, cache=None, num_pred=None, keep_all_weights=False):
        device = features_1d.device

        # Token to Embedding
        pos_tokens = self.emb(tokens).permute(0, 2, 1)

        # Add 1D Positional Encoding
        pos_tokens = self.pe_1d(pos_tokens, start=start)
        pos_tokens = pos_tokens.permute(2, 0, 1)

        if num_pred is None:
            num_pred = tokens.size(1)

        # Use cache values to avoid useless computation at eval time
        if self.dec_att_win > 1 and cache is not None:
                cache = cache[:, -self.dec_att_win+1:]
        else:
            cache = None
        max_len = max(token_len)
        num_tokens_to_keep = num_pred if self.dec_att_win is None else min([num_pred + self.dec_att_win - 1, pos_tokens.size(0), max_len])
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        # Generate dynamic masks
        target_mask = self.generate_target_mask(tokens.size(1), device)  # Use only already predicted tokens (causal)
        memory_mask = None  # Use all feature position

        # Generate static masks
        key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)  # Use all token except padding
        key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)  # Use all feature position except padding

        target_mask = target_mask[-num_pred:, -num_tokens_to_keep:]
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
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3])
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

    def generate_token_mask(self, token_len, total_size, device):
        """
        Generate mask for tokens per sample
        Fake mask: real masking is handled by loss (attention window causes nan)
        """
        batch_size, len_max = total_size
        mask = torch.zeros((batch_size, len_max), dtype=torch.bool, device=device)
        return mask

    def generate_target_mask(self, target_len, device):
        """
        Generate mask for tokens per time step (teacher forcing)
        """
        if self.dec_att_win == 1:
            return torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=1)
        else:
            return torch.logical_not(
                torch.logical_and(torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                                  torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=-self.dec_att_win+1)))

    def get_pos_features(self, features):
        return self.pe_2d(features)
