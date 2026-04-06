import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from android.et.et_export import (  # noqa: E402
    DEFAULT_GPT2_PRETRAINED_DIR,
    DemoConfig,
    build_training_module,
    export_training_program,
    iter_batches,
    prepare_wikitext_token_files,
)


DEFAULT_DEVICE_WORKSPACE = "/data/local/tmp/mobizo_gpt2_training"
DEFAULT_WIKITEXT_RAW_DIR = str(REPO_ROOT / "data" / "wikitext2" / "wikitext-2-raw")
RUNNER_SOURCE_DIR = REPO_ROOT / "android" / "et" / "training_runner"


def run(cmd, cwd=None, capture_output=False):
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def adb_prefix(device: str):
    return ["adb", "-s", device]


def default_artifact_path(artifact_dir: Path, model_preset: str) -> Path:
    if model_preset == "gpt2_small":
        return artifact_dir / "gpt2_small_training.pte"
    return artifact_dir / "gpt2_tiny_training.pte"


def maybe_prepare_token_data(args, artifact_dir: Path) -> tuple[str, str]:
    train_tokens_path = args.train_tokens_path
    eval_tokens_path = args.eval_tokens_path
    if train_tokens_path and (eval_tokens_path or not args.prepare_data):
        return train_tokens_path, eval_tokens_path

    if not args.prepare_data:
        return train_tokens_path, eval_tokens_path

    raw_dir = args.raw_data_dir
    prepare_dir = artifact_dir / "prepared_tokens"
    token_paths = prepare_wikitext_token_files(
        raw_dir=raw_dir,
        tokenizer_dir=args.pretrained_dir,
        out_dir=str(prepare_dir),
    )
    train_tokens_path = train_tokens_path or token_paths.get("train", "")
    eval_tokens_path = eval_tokens_path or token_paths.get("valid", "")
    return train_tokens_path, eval_tokens_path


def build_runner(args, runner_build_dir: Path):
    android_ndk_root = os.environ["ANDROID_NDK_ROOT"]
    executorch_root = Path(os.environ["EXECUTORCH_ROOT"])
    configure_cmd = [
        "cmake",
        "-S",
        str(RUNNER_SOURCE_DIR),
        "-B",
        str(runner_build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_TOOLCHAIN_FILE={android_ndk_root}/build/cmake/android.toolchain.cmake",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-30",
        f"-DEXECUTORCH_ROOT={executorch_root}",
        f"-DEXECUTORCH_ANDROID_BUILD_DIR={args.build_folder}",
    ]
    build_cmd = ["cmake", "--build", str(runner_build_dir), "-j8"]
    run(configure_cmd, cwd=REPO_ROOT)
    run(build_cmd, cwd=REPO_ROOT)


def export_artifact(args, artifact_path: Path) -> list[str]:
    cfg = DemoConfig(
        model_preset=args.model_preset,
        pretrained_dir=args.pretrained_dir,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        num_batches=max(args.steps, args.eval_steps, 1),
        learning_rate=args.learning_rate,
        steps=args.steps,
        eval_steps=args.eval_steps,
        train_tokens_path=args.train_tokens_path,
        eval_tokens_path=args.eval_tokens_path,
        dataset_stride=args.dataset_stride,
        untie_lm_head_for_training_export=args.untie_lm_head_for_training_export,
        emit_predictions=not args.loss_only,
    )
    module = build_training_module(cfg)
    example_inputs = next(iter(iter_batches(cfg, cfg.train_tokens_path)))
    return export_training_program(
        module,
        example_inputs,
        str(artifact_path),
        use_xnnpack=False,
        external_constants=args.external_constants,
    )


def device_run(args, runner_path: Path, artifact_path: Path, artifact_dir: Path):
    prefix = adb_prefix(args.device)
    run(prefix + ["shell", f"rm -rf {args.device_workspace}"])
    run(prefix + ["shell", f"mkdir -p {args.device_workspace}"])

    remote_runner = f"{args.device_workspace}/gpt2_training_runner"
    remote_model = f"{args.device_workspace}/{artifact_path.name}"
    remote_ptd = (
        f"{args.device_workspace}/{Path(args.ptd_path).name}" if args.ptd_path else None
    )
    remote_save_ptd = (
        f"{args.device_workspace}/{Path(args.save_ptd_path).name}"
        if args.save_ptd_path
        else None
    )
    remote_train_tokens = (
        f"{args.device_workspace}/{Path(args.train_tokens_path).name}"
        if args.train_tokens_path
        else None
    )
    remote_eval_tokens = (
        f"{args.device_workspace}/{Path(args.eval_tokens_path).name}"
        if args.eval_tokens_path
        else None
    )

    run(prefix + ["push", str(runner_path), remote_runner])
    run(prefix + ["push", str(artifact_path), remote_model])
    if args.ptd_path:
        run(prefix + ["push", args.ptd_path, remote_ptd])
    if args.train_tokens_path:
        run(prefix + ["push", args.train_tokens_path, remote_train_tokens])
    if args.eval_tokens_path:
        run(prefix + ["push", args.eval_tokens_path, remote_eval_tokens])

    cmd_parts = [
        f"cd {shlex.quote(args.device_workspace)}",
        "chmod +x ./gpt2_training_runner",
        "./gpt2_training_runner",
        f"--model_path {shlex.quote(remote_model)}",
        f"--method_name {shlex.quote(args.method_name)}",
        f"--steps {args.steps}",
        f"--eval_steps {args.eval_steps}",
        f"--batch_size {args.batch_size}",
        f"--block_size {args.block_size}",
        f"--vocab_size {args.vocab_size}",
        f"--dataset_stride {args.dataset_stride}",
        f"--learning_rate {args.learning_rate}",
        f"--warmup_steps {args.warmup_steps}",
        f"--log_every {args.log_every}",
        f"--seed {args.seed}",
    ]
    if remote_ptd:
        cmd_parts.append(f"--ptd_path {shlex.quote(remote_ptd)}")
    if remote_save_ptd:
        cmd_parts.append(f"--save_ptd_path {shlex.quote(remote_save_ptd)}")
    if remote_train_tokens:
        cmd_parts.append(f"--train_tokens_path {shlex.quote(remote_train_tokens)}")
    if remote_eval_tokens:
        cmd_parts.append(f"--eval_tokens_path {shlex.quote(remote_eval_tokens)}")
    if args.tie_embedding_head:
        cmd_parts.append("--tie_embedding_head")

    shell_cmd = " && ".join(cmd_parts[:2]) + " && " + " ".join(cmd_parts[2:])
    result = run(prefix + ["shell", shell_cmd], capture_output=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    log_path = artifact_dir / "device_training.log"
    log_path.write_text(result.stdout)
    if remote_save_ptd:
        run(
            prefix
            + ["pull", remote_save_ptd, str(artifact_dir / Path(args.save_ptd_path).name)]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export and run the ExecuTorch GPT2 CPU training baseline on Android."
    )
    parser.add_argument("-s", "--device", required=True, help="ADB serial number.")
    parser.add_argument(
        "-b",
        "--build_folder",
        required=True,
        help="ExecuTorch Android build directory, e.g. $EXECUTORCH_ROOT/build-android-qnn237",
    )
    parser.add_argument(
        "-a",
        "--artifact_dir",
        default="./gpt2_training_phone_baseline",
        help="Local directory for exported artifacts and logs.",
    )
    parser.add_argument(
        "--artifact_path",
        default="",
        help="Reuse an existing training PTE instead of exporting a new one.",
    )
    parser.add_argument(
        "--runner_build_dir",
        default="./android/et/training_runner/build-android",
        help="Local CMake build directory for the native runner.",
    )
    parser.add_argument(
        "--model_preset",
        choices=("tiny", "gpt2_small"),
        default="tiny",
    )
    parser.add_argument(
        "--pretrained_dir",
        default=DEFAULT_GPT2_PRETRAINED_DIR,
        help="Local HuggingFace GPT-2 checkpoint directory.",
    )
    parser.add_argument(
        "--raw_data_dir",
        default=DEFAULT_WIKITEXT_RAW_DIR,
        help="Directory containing wiki.train.raw/wiki.valid.raw/wiki.test.raw.",
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Tokenize raw WikiText into int32 bins under the artifact directory.",
    )
    parser.add_argument("--train_tokens_path", default="")
    parser.add_argument("--eval_tokens_path", default="")
    parser.add_argument("--method_name", default="forward")
    parser.add_argument("--ptd_path", default="")
    parser.add_argument("--save_ptd_path", default="")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--dataset_stride", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--external_constants", action="store_true")
    parser.add_argument("--untie_lm_head_for_training_export", action="store_true")
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--tie_embedding_head", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_workspace", default=DEFAULT_DEVICE_WORKSPACE)
    args = parser.parse_args()

    if args.model_preset == "gpt2_small":
        if args.block_size == 16:
            args.block_size = 128
        if args.batch_size == 8:
            args.batch_size = 1
        if args.learning_rate == 0.05:
            args.learning_rate = 1e-4
        args.external_constants = True
        args.untie_lm_head_for_training_export = True
        args.loss_only = True
        args.tie_embedding_head = True
        if not args.prepare_data and not args.train_tokens_path:
            args.prepare_data = True

    artifact_dir = (REPO_ROOT / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.train_tokens_path, args.eval_tokens_path = maybe_prepare_token_data(
        args, artifact_dir
    )

    artifact_path = (
        Path(args.artifact_path).resolve()
        if args.artifact_path
        else default_artifact_path(artifact_dir, args.model_preset)
    )
    runner_build_dir = (REPO_ROOT / args.runner_build_dir).resolve()
    runner_path = runner_build_dir / "gpt2_training_runner"

    exported_ptd_paths: list[str] = []
    if not args.skip_export:
        exported_ptd_paths = export_artifact(args, artifact_path)

    if not args.ptd_path and exported_ptd_paths:
        args.ptd_path = exported_ptd_paths[0]

    if not args.skip_build:
        build_runner(args, runner_build_dir)

    device_run(args, runner_path, artifact_path, artifact_dir)


if __name__ == "__main__":
    main()
