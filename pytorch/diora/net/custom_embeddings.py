import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from diora.data.treeencoding import BatchMap, CharBatchMap
from diora.net.parser_types import Embedding

from typing import List, Protocol
from typing_extensions import override

import math


class WordEmbedding(Embedding[BatchMap]):
    ...


class CharEmbedding(Embedding[CharBatchMap]):
    ...


class SimpleEmbedding(nn.Embedding, WordEmbedding):
    def __init__(self, num_words : int, out_dim : int):
        super().__init__(num_words, out_dim)
    
    def forward(self, batch : BatchMap) -> torch.Tensor:
        return super().forward(batch["input_ids"])
    

class CharEmbed(CharEmbedding):
    def __init__(self, num_chars : int, out_dim : int):
        super().__init__()

        self.embed : nn.Embedding = nn.Embedding(num_chars, out_dim)
    
    def forward(self, batch : CharBatchMap) -> List[List[torch.Tensor]]:
        return [[self.embed(word) for word in sentence] for sentence in batch["char_ids"]]


class DualEmbed(CharEmbedding):
    """Produces char embeds only based on
    known tokens."""
    def __init__(self, num_words, num_chars, word_emb_dim : int, char_emb_dim : int):
        super().__init__()

        self.word_emd_dim : int = word_emb_dim
        self.char_emb_dim : int = char_emb_dim

        self.word_embedding : SimpleEmbedding = SimpleEmbedding(num_words, word_emb_dim)

        self.char_embedder : CharEmbed = CharEmbed(num_chars, char_emb_dim)

        self.char_lstm : nn.LSTM = nn.LSTM(char_emb_dim, char_emb_dim // 2, 1, bidirectional = True, batch_first = True)
        
    def forward(self, batch : CharBatchMap) -> torch.Tensor:
        sentence_length : int = batch["input_ids"].shape[1]

        word_embeddings : torch.Tensor = self.word_embedding(batch) # B x S x E
        char_embeddings : List[List[torch.Tensor]] = self.char_embedder(batch) # B x S x Ws x E

        char_embeds_flat : List[torch.Tensor] = [word for sentence in char_embeddings for word in sentence] # B*S x Ws x E

        packed_sequence : torch.nn.utils.rnn.PackedSequence 
        packed_sequence = pack_sequence(char_embeds_flat, enforce_sorted=False)

        bi_encoded : torch.Tensor
        _, (bi_encoded, _) = self.char_lstm(packed_sequence)     # 2 x B*S x E//2

        bi_encoded = torch.cat((bi_encoded[0], bi_encoded[1]), dim = 1) # B*S x E

        bi_encoded = bi_encoded.view(-1, sentence_length, self.char_emb_dim)    # B x S x E

        return torch.cat((word_embeddings, bi_encoded), dim = 2) # B x S x 2E  # TODO
    

class DualTransformerEmbed(CharEmbedding):
    """Produces char embeds only based on
    known tokens."""
    def __init__(self, num_words, num_chars, word_emb_dim : int, char_emb_dim : int):
        super().__init__()

        self.word_emd_dim : int = word_emb_dim
        self.char_emb_dim : int = char_emb_dim

        self.word_embedding : SimpleEmbedding = SimpleEmbedding(num_words, word_emb_dim)

        self.char_embedder : CharEmbed = CharEmbed(num_chars, char_emb_dim)

        self.char_attention_1 : nn.MultiheadAttention = nn.MultiheadAttention(embed_dim = char_emb_dim,
                                                                              num_heads = 4,
                                                                              batch_first = True)
        
        self.char_attention_2 : nn.MultiheadAttention = nn.MultiheadAttention(embed_dim = char_emb_dim,
                                                                              num_heads = 4,
                                                                              batch_first = True)

        self.pos_encoder = PositionalEncoding(char_emb_dim, 0.0)

        self.norm = nn.LayerNorm(self.char_emb_dim)

        self.word_target : nn.Parameter = nn.Parameter(torch.rand(char_emb_dim), requires_grad = True)

    def forward(self, batch : CharBatchMap) -> torch.Tensor:

        batch_size : int = batch["batch_size"]

        word_embeddings : torch.Tensor = self.word_embedding(batch) # B x S x E
        char_embeddings : List[List[torch.Tensor]] = self.char_embedder(batch) # B x S x Ws x E

        char_embeds_flat : List[torch.Tensor] = [word for sentence in char_embeddings for word in sentence] # B*S x Ws x E
        lengths : torch.Tensor = torch.tensor([word.shape[0] for word in char_embeds_flat], device = word_embeddings.device)

        padded : torch.Tensor = pad_sequence(char_embeds_flat, batch_first = True) # B*S x W x E

        w_range : torch.Tensor = torch.range(0, padded.shape[1]-1, device = padded.device).repeat(padded.shape[0], 1)

        mask : torch.Tensor = w_range >= lengths.repeat(w_range.shape[1], 1).T


        padded = padded * math.sqrt(self.char_emb_dim)
        padded = self.pos_encoder(padded)

        transformed : torch.Tensor 
        transformed, _ = self.char_attention_1(padded, padded, padded, key_padding_mask = mask) # B*S x W x E
        transformed = self.norm(transformed)

        target : torch.Tensor = self.word_target.repeat(transformed.shape[0], 1, 1) # B*S x 1 x E

        transformed, _ = self.char_attention_2(target, padded, padded, key_padding_mask = mask) # B*S x 1 x E

        transformed = transformed.squeeze(1) # B*S x E

        transformed = transformed.view(batch_size, -1, self.char_emb_dim) # B x S x E
        transformed = self.norm(transformed)

        return torch.cat((word_embeddings, transformed), dim = 2) # B x S x 2E  # TODO
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
