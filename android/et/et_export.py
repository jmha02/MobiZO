import argparse
import math
import os
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.export import export
from torch.export.experimental import _export_forward_backward

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from android.models.gpt2 import (  # noqa: E402
    GPT2Config,
    GPT2Model,
    load_pretrained_gpt2_weights,
    make_gpt2_small_config,
)


DEFAULT_ARTIFACT_PATH = "./et_training_artifacts/gpt2_tiny_training.pte"
DEFAULT_GPT2_PRETRAINED_DIR = (
    "/home/jmha/MobileFineTuner/gpt2_lora_finetune/pretrained/gpt2"
)


@dataclass
class DemoConfig:
    model_preset: str = "tiny"
    pretrained_dir: str = ""
    vocab_size: int = 32
    block_size: int = 16
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    batch_size: int = 8
    num_batches: int = 64
    learning_rate: float = 0.05
    steps: int = 50
    eval_steps: int = 8
    train_tokens_path: str = ""
    eval_tokens_path: str = ""
    dataset_stride: int = 0
    untie_lm_head_for_training_export: bool = False
    emit_predictions: bool = True


class GPT2TrainingModule(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        ignore_index: int = -100,
        emit_predictions: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.emit_predictions = emit_predictions
        self.register_buffer(
            "ignore_labels_cache",
            torch.full((1, 1), ignore_index, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self, tokens: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(tokens)
        ignore_tail = self.ignore_labels_cache.expand(labels.shape[0], 1)
        shifted_labels = torch.hstack((labels[..., 1:], ignore_tail))
        loss = self.loss(
            logits.reshape(-1, logits.size(-1)),
            shifted_labels.reshape(-1),
        )
        if not self.emit_predictions:
            return (loss,)
        predictions = logits.detach().argmax(dim=-1)
        return loss, predictions


def build_tokenizer(tokenizer_dir: str):
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _token_bin_name(split: str) -> str:
    return f"wiki.{split}.gpt2.int32.bin"


def prepare_wikitext_token_file(raw_path: str, tokenizer_dir: str, out_path: str) -> int:
    tokenizer = build_tokenizer(tokenizer_dir)
    tokens = array("i")
    if tokens.itemsize != 4:
        raise RuntimeError("Expected 4-byte int array for token serialization.")

    with open(raw_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if line:
                tokens.extend(tokenizer.encode(line, add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as file:
        tokens.tofile(file)
    return len(tokens)


def prepare_wikitext_token_files(
    raw_dir: str, tokenizer_dir: str, out_dir: str
) -> dict[str, str]:
    mapping = {
        "train": "wiki.train.raw",
        "valid": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }
    token_paths: dict[str, str] = {}
    for split, raw_name in mapping.items():
        raw_path = Path(raw_dir) / raw_name
        if not raw_path.exists():
            continue
        token_path = Path(out_dir) / _token_bin_name(split)
        num_tokens = prepare_wikitext_token_file(
            str(raw_path), tokenizer_dir, str(token_path)
        )
        print(f"Prepared {token_path} with {num_tokens} tokens")
        token_paths[split] = str(token_path)
    return token_paths


def make_synthetic_batch(
    batch_size: int, block_size: int, vocab_size: int, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    start = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1) + offset
    positions = torch.arange(block_size, dtype=torch.int64).unsqueeze(0)
    tokens = (start + positions) % vocab_size
    return tokens, tokens.clone()


def load_token_tensor(token_path: str) -> torch.Tensor:
    tokens = array("i")
    if tokens.itemsize != 4:
        raise RuntimeError("Expected 4-byte int array for token deserialization.")
    with open(token_path, "rb") as file:
        tokens.frombytes(file.read())
    return torch.tensor(tokens, dtype=torch.int64)


def make_token_batch(
    token_tensor: torch.Tensor,
    batch_size: int,
    block_size: int,
    offset: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_tensor.numel() < block_size:
        raise ValueError(
            f"Token corpus is too short: {token_tensor.numel()} < block_size {block_size}"
        )

    stride = stride if stride > 0 else block_size
    max_start = token_tensor.numel() - block_size
    num_windows = max_start // stride + 1
    starts = (
        (offset * batch_size + torch.arange(batch_size, dtype=torch.int64)) % num_windows
    ) * stride
    batch = torch.stack(
        [token_tensor[start : start + block_size] for start in starts.tolist()],
        dim=0,
    )
    return batch, batch.clone()


def iter_token_batches(
    cfg: DemoConfig, token_path: str
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    token_tensor = load_token_tensor(token_path)
    for batch_idx in range(cfg.num_batches):
        yield make_token_batch(
            token_tensor=token_tensor,
            batch_size=cfg.batch_size,
            block_size=cfg.block_size,
            offset=batch_idx,
            stride=cfg.dataset_stride,
        )


def iter_synthetic_batches(
    cfg: DemoConfig,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for batch_idx in range(cfg.num_batches):
        yield make_synthetic_batch(
            batch_size=cfg.batch_size,
            block_size=cfg.block_size,
            vocab_size=cfg.vocab_size,
            offset=batch_idx,
        )


def iter_batches(
    cfg: DemoConfig, token_path: str = ""
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if token_path:
        return iter_token_batches(cfg, token_path)
    return iter_synthetic_batches(cfg)


def build_training_module(cfg: DemoConfig) -> GPT2TrainingModule:
    if cfg.model_preset == "gpt2_small":
        if not cfg.pretrained_dir:
            raise ValueError("pretrained_dir is required for model_preset=gpt2_small")
        model_cfg = make_gpt2_small_config(block_size=cfg.block_size)
        model_cfg.tie_word_embeddings = not cfg.untie_lm_head_for_training_export
        module = GPT2TrainingModule(
            GPT2Model(model_cfg), emit_predictions=cfg.emit_predictions
        )
        load_pretrained_gpt2_weights(module.model, cfg.pretrained_dir)
        return module

    model_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=0.0,
        bias=True,
    )
    return GPT2TrainingModule(GPT2Model(model_cfg), emit_predictions=cfg.emit_predictions)


def export_training_program(
    module: GPT2TrainingModule,
    example_inputs: tuple[torch.Tensor, torch.Tensor],
    out_path: str,
    use_xnnpack: bool,
    external_constants: bool = False,
) -> list[str]:
    from executorch.exir import to_edge
    from executorch.exir.capture import ExecutorchBackendConfig

    exported = export(module.eval(), example_inputs, strict=False)
    joint_graph = _export_forward_backward(exported)
    edge_program = to_edge(joint_graph)

    if use_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        edge_program = edge_program.to_backend(
            XnnpackPartitioner(force_fp32_dynamic_linear=True)
        )

    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(external_constants=external_constants)
    )
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as file:
        file.write(executorch_program.buffer)

    existing_ptd = set(Path(out_dir).glob("*.ptd"))
    executorch_program.write_tensor_data_to_file(out_dir)
    current_ptd = set(Path(out_dir).glob("*.ptd"))
    written_ptd = sorted(str(path) for path in current_ptd - existing_ptd)
    if external_constants and not written_ptd:
        written_ptd = sorted(str(path) for path in current_ptd)
    return written_ptd


def evaluate_with_portable(et_mod, cfg: DemoConfig, num_eval_steps: int) -> float:
    total_loss = 0.0
    eval_token_path = cfg.eval_tokens_path or cfg.train_tokens_path
    for step, batch in enumerate(iter_batches(cfg, eval_token_path)):
        if step >= num_eval_steps:
            break
        loss = et_mod.forward(batch)[0]
        total_loss += float(loss)
    return total_loss / num_eval_steps


def train_with_training_module(
    model_bytes: bytes, cfg: DemoConfig
) -> dict[str, float]:
    from executorch.extension.training import (
        _load_for_executorch_for_training_from_buffer,
        get_sgd_optimizer,
    )

    module = _load_for_executorch_for_training_from_buffer(model_bytes)
    init_batch = next(iter(iter_batches(cfg, cfg.train_tokens_path)))
    module.forward_backward("forward", init_batch)
    optimizer = get_sgd_optimizer(module.named_parameters(), cfg.learning_rate)

    pre_loss = evaluate_with_portable(module.model, cfg, cfg.eval_steps)
    losses = []
    for step, batch in enumerate(iter_batches(cfg, cfg.train_tokens_path)):
        if step >= cfg.steps:
            break
        outputs = module.forward_backward("forward", batch)
        losses.append(float(outputs[0]))
        optimizer.step(module.named_gradients())
    post_loss = evaluate_with_portable(module.model, cfg, cfg.eval_steps)
    return {
        "pre_eval_loss": pre_loss,
        "post_eval_loss": post_loss,
        "last_train_loss": losses[-1],
    }


def train_with_portable_fallback(
    model_path: str, cfg: DemoConfig, ptd_path: str = ""
) -> dict[str, float]:
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    et_mod = _load_for_executorch(model_path, ptd_path or None)
    grad_start = et_mod.run_method("__et_training_gradients_index_forward", [])[0]
    param_start = et_mod.run_method("__et_training_parameters_index_forward", [])[0]

    pre_loss = evaluate_with_portable(et_mod, cfg, cfg.eval_steps)
    losses = []
    for step, batch in enumerate(iter_batches(cfg, cfg.train_tokens_path)):
        if step >= cfg.steps:
            break
        outputs = et_mod.forward(batch, clone_outputs=False)
        losses.append(float(outputs[0]))
        with torch.no_grad():
            for grad, param in zip(outputs[grad_start:param_start], outputs[param_start:]):
                param.sub_(cfg.learning_rate * grad)
    post_loss = evaluate_with_portable(et_mod, cfg, cfg.eval_steps)
    return {
        "pre_eval_loss": pre_loss,
        "post_eval_loss": post_loss,
        "last_train_loss": losses[-1],
    }


def run_training(
    model_path: str, cfg: DemoConfig, ptd_path: str = ""
) -> dict[str, float]:
    with open(model_path, "rb") as file:
        model_bytes = file.read()

    if ptd_path:
        print("External constants detected, using portable fallback for host training.")
        return train_with_portable_fallback(model_path, cfg, ptd_path)

    try:
        return train_with_training_module(model_bytes, cfg)
    except Exception as exc:
        print(f"TrainingModule path unavailable, falling back to portable runner: {exc}")
        return train_with_portable_fallback(model_path, cfg)


def default_artifact_name(model_preset: str) -> str:
    if model_preset == "gpt2_small":
        return "./et_training_artifacts/gpt2_small_training.pte"
    return DEFAULT_ARTIFACT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT2 ExecuTorch joint-graph export and training demo."
    )
    parser.add_argument(
        "--model_preset",
        choices=("tiny", "gpt2_small"),
        default="tiny",
    )
    parser.add_argument(
        "--artifact",
        default="",
        help="Output PTE path.",
    )
    parser.add_argument(
        "--pretrained_dir",
        default=DEFAULT_GPT2_PRETRAINED_DIR,
        help="Local HuggingFace GPT-2 checkpoint directory.",
    )
    parser.add_argument(
        "--train_tokens_path",
        default="",
        help="Optional pretokenized int32 train token file.",
    )
    parser.add_argument(
        "--eval_tokens_path",
        default="",
        help="Optional pretokenized int32 eval token file.",
    )
    parser.add_argument(
        "--prepare_raw_dir",
        default="",
        help="If set, tokenize wiki.{train,valid,test}.raw under this directory.",
    )
    parser.add_argument(
        "--prepare_data_dir",
        default="",
        help="Output directory for pretokenized dataset files.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--dataset_stride", type=int, default=0)
    parser.add_argument(
        "--untie_lm_head_for_training_export",
        action="store_true",
        help="Duplicate the GPT-2 LM head instead of tying it to token embeddings.",
    )
    parser.add_argument(
        "--loss_only",
        action="store_true",
        help="Export only the loss output, without argmax predictions.",
    )
    parser.add_argument(
        "--external_constants",
        action="store_true",
        help="Store constant weights in external .ptd files.",
    )
    parser.add_argument(
        "--no_xnnpack",
        action="store_true",
        help="Disable XNNPACK partitioning during export.",
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip export and reuse an existing PTE.",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Export the training PTE and exit without running host-side training.",
    )
    args = parser.parse_args()

    if not args.artifact:
        args.artifact = default_artifact_name(args.model_preset)

    if args.prepare_raw_dir:
        prepare_dir = args.prepare_data_dir or str(
            REPO_ROOT / "data" / "wikitext2" / "gpt2_tokens"
        )
        token_paths = prepare_wikitext_token_files(
            raw_dir=args.prepare_raw_dir,
            tokenizer_dir=args.pretrained_dir,
            out_dir=prepare_dir,
        )
        if not args.train_tokens_path and "train" in token_paths:
            args.train_tokens_path = token_paths["train"]
        if not args.eval_tokens_path and "valid" in token_paths:
            args.eval_tokens_path = token_paths["valid"]

    if args.model_preset == "gpt2_small":
        if args.block_size == 16:
            args.block_size = 128
        if args.batch_size == 8:
            args.batch_size = 1
        if args.learning_rate == 0.05:
            args.learning_rate = 1e-4
        if not args.untie_lm_head_for_training_export:
            args.untie_lm_head_for_training_export = True
        if not args.loss_only:
            args.loss_only = True

    cfg = DemoConfig(
        model_preset=args.model_preset,
        pretrained_dir=args.pretrained_dir,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        num_batches=max(args.num_batches, args.steps, args.eval_steps, 1),
        learning_rate=args.learning_rate,
        steps=args.steps,
        eval_steps=args.eval_steps,
        train_tokens_path=args.train_tokens_path,
        eval_tokens_path=args.eval_tokens_path,
        dataset_stride=args.dataset_stride,
        untie_lm_head_for_training_export=args.untie_lm_head_for_training_export,
        emit_predictions=not args.loss_only,
    )

    torch.manual_seed(0)
    module = build_training_module(cfg)
    example_inputs = next(iter(iter_batches(cfg, cfg.train_tokens_path)))

    ptd_paths: list[str] = []
    if not args.skip_export:
        ptd_paths = export_training_program(
            module,
            example_inputs,
            args.artifact,
            use_xnnpack=not args.no_xnnpack,
            external_constants=args.external_constants,
        )

    if args.export_only:
        print(f"Exported training artifact: {args.artifact}")
        for path in ptd_paths:
            print(f"Exported external data: {path}")
        return

    ptd_path = ptd_paths[0] if ptd_paths else ""
    stats = run_training(args.artifact, cfg, ptd_path=ptd_path)
    print("Training stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    print(
        "  improvement:",
        f"{stats['pre_eval_loss'] - stats['post_eval_loss']:.6f}",
    )


if __name__ == "__main__":
    main()
