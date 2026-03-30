import math

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch import nn


@dataclass
class GPT2Config:
    block_size: int = 128
    vocab_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        mask = torch.full(
            (1, 1, config.block_size, config.block_size), float("-inf")
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, channels = x.shape

        q = self.q_proj(x).view(batch_size, seqlen, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seqlen, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seqlen, self.n_head, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, channels)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    @staticmethod
    def new_gelu(x: torch.Tensor) -> torch.Tensor:
        # Avoid lowering through a dedicated GELU op on older QNN paths.
        coeff = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * x * x * x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.new_gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Backbone(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Identity()
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.register_buffer(
            "position_ids",
            torch.arange(config.block_size, dtype=torch.int32),
            persistent=False,
        )

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        seqlen = tokens.shape[1]
        positions = self.position_ids[:seqlen]
        x = self.wte(tokens) + self.wpe(positions)
        return self.drop(x)

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decode(self.embed(tokens))


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.transformer = GPT2Backbone(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(tokens)
        return self.lm_head(hidden_states)


class GPT2EmbeddingModel(nn.Module):
    def __init__(self, transformer: GPT2Backbone):
        super().__init__()
        self.wte = transformer.wte
        self.wpe = transformer.wpe
        self.drop = transformer.drop
        self.register_buffer(
            "position_ids", transformer.position_ids, persistent=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seqlen = tokens.shape[1]
        positions = self.position_ids[:seqlen]
        return self.drop(self.wte(tokens) + self.wpe(positions))


class GPT2DecoderModel(nn.Module):
    def __init__(self, transformer: GPT2Backbone, lm_head: nn.Linear):
        super().__init__()
        self.h = transformer.h
        self.ln_f = transformer.ln_f
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
