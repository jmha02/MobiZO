import math
from pathlib import Path

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from safetensors import safe_open
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
    tie_word_embeddings: bool = False


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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        else:
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


def make_gpt2_small_config(block_size: int = 128) -> GPT2Config:
    return GPT2Config(
        block_size=block_size,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        tie_word_embeddings=True,
    )


def load_pretrained_gpt2_weights(
    model: GPT2Model, pretrained_dir: str | Path
) -> GPT2Model:
    pretrained_dir = Path(pretrained_dir)
    checkpoint_path = pretrained_dir / "model.safetensors"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = model.transformer.config
    if config.tie_word_embeddings and model.lm_head.weight is not model.transformer.wte.weight:
        raise ValueError("Expected tied word embeddings for pretrained GPT-2 load.")

    with torch.no_grad(), safe_open(
        str(checkpoint_path), framework="pt", device="cpu"
    ) as checkpoint:
        position_weight = checkpoint.get_tensor("wpe.weight")
        if config.block_size > position_weight.shape[0]:
            raise ValueError(
                f"block_size {config.block_size} exceeds pretrained position embeddings "
                f"{position_weight.shape[0]}"
            )

        model.transformer.wte.weight.copy_(checkpoint.get_tensor("wte.weight"))
        model.transformer.wpe.weight.copy_(position_weight[: config.block_size])

        for layer_idx, block in enumerate(model.transformer.h):
            prefix = f"h.{layer_idx}"

            block.ln_1.weight.copy_(checkpoint.get_tensor(f"{prefix}.ln_1.weight"))
            block.ln_1.bias.copy_(checkpoint.get_tensor(f"{prefix}.ln_1.bias"))
            block.ln_2.weight.copy_(checkpoint.get_tensor(f"{prefix}.ln_2.weight"))
            block.ln_2.bias.copy_(checkpoint.get_tensor(f"{prefix}.ln_2.bias"))

            attn_weight = checkpoint.get_tensor(f"{prefix}.attn.c_attn.weight")
            q_weight, k_weight, v_weight = attn_weight.split(config.n_embd, dim=1)
            block.attn.q_proj.weight.copy_(q_weight.t().contiguous())
            block.attn.k_proj.weight.copy_(k_weight.t().contiguous())
            block.attn.v_proj.weight.copy_(v_weight.t().contiguous())

            attn_bias = checkpoint.get_tensor(f"{prefix}.attn.c_attn.bias")
            q_bias, k_bias, v_bias = attn_bias.split(config.n_embd, dim=0)
            block.attn.q_proj.bias.copy_(q_bias)
            block.attn.k_proj.bias.copy_(k_bias)
            block.attn.v_proj.bias.copy_(v_bias)

            block.attn.c_proj.weight.copy_(
                checkpoint.get_tensor(f"{prefix}.attn.c_proj.weight").t().contiguous()
            )
            block.attn.c_proj.bias.copy_(
                checkpoint.get_tensor(f"{prefix}.attn.c_proj.bias")
            )

            block.mlp.c_fc.weight.copy_(
                checkpoint.get_tensor(f"{prefix}.mlp.c_fc.weight").t().contiguous()
            )
            block.mlp.c_fc.bias.copy_(checkpoint.get_tensor(f"{prefix}.mlp.c_fc.bias"))
            block.mlp.c_proj.weight.copy_(
                checkpoint.get_tensor(f"{prefix}.mlp.c_proj.weight").t().contiguous()
            )
            block.mlp.c_proj.bias.copy_(
                checkpoint.get_tensor(f"{prefix}.mlp.c_proj.bias")
            )

        model.transformer.ln_f.weight.copy_(checkpoint.get_tensor("ln_f.weight"))
        model.transformer.ln_f.bias.copy_(checkpoint.get_tensor("ln_f.bias"))
        if not config.tie_word_embeddings:
            model.lm_head.weight.copy_(checkpoint.get_tensor("wte.weight"))

    return model
