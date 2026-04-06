# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import sys
from multiprocessing.connection import Client

import numpy as np
import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DTYPE,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)

from utils import (
    build_executorch_binary,
    make_quantizer,
    make_llm_quantizer,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
    parse_skip_delegation_node,
)


def create_device_inputs(example_inputs, artifact_dir):
    inputs = [inp for inp in example_inputs]
    input_list = ""
    for i, inp in enumerate(inputs):
        inp.detach().cpu().numpy().tofile(f"{artifact_dir}/input_{i}_0.raw")
        input_list += f"input_{i}_0.raw\n"

    return tuple(inputs), input_list


def collect_outputs(output_folder, dtype=np.float32):
    output_raws = []
    for f in sorted(os.listdir(output_folder), key=lambda f: int(f.split("_")[1])):
        filename = os.path.join(output_folder, f)
        if re.match(r"^output_[0-9]+_[1-9].raw$", f):
            os.remove(filename)
        elif f.endswith(".raw"):
            output_raws.append(np.fromfile(filename, dtype=dtype))
    return output_raws


def torch_dtype_to_numpy(dtype):
    mapping = {
        torch.uint8: np.uint8,
        torch.uint16: np.uint16,
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype conversion for {dtype}")
    return mapping[dtype]


def quantize_array(array, encoding):
    if encoding is None:
        return array
    q = np.round(array / float(encoding[QCOM_SCALE]) + int(encoding[QCOM_ZERO_POINT]))
    q = np.clip(q, int(encoding[QCOM_QUANT_MIN]), int(encoding[QCOM_QUANT_MAX]))
    return q.astype(torch_dtype_to_numpy(encoding[QCOM_DTYPE]))


def dequantize_array(array, encoding):
    if encoding is None:
        return array.astype(np.float32)
    return (
        array.astype(np.float32) - int(encoding[QCOM_ZERO_POINT])
    ) * float(encoding[QCOM_SCALE])


def run_pte_on_device(
    args,
    pte_path,
    artifact_dir,
    device_inputs,
    workspace_name,
    input_encodings=None,
    output_encodings=None,
):
    os.makedirs(artifact_dir, exist_ok=True)
    if input_encodings is not None:
        quantized_inputs = []
        for inp, encoding in zip(device_inputs, input_encodings):
            quantized_array = quantize_array(inp.detach().cpu().numpy(), encoding)
            quantized_inputs.append(torch.from_numpy(quantized_array))
        device_inputs = tuple(quantized_inputs)
    input_tensors, input_list = create_device_inputs(device_inputs, artifact_dir)

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{workspace_name}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=input_tensors, input_list=input_list)
    adb.execute()

    output_data_folder = f"{artifact_dir}/outputs"
    make_output_dir(output_data_folder)
    adb.pull(output_path=artifact_dir)

    output_dtype = np.float32
    if output_encodings:
        first_output_encoding = output_encodings[0]
        if first_output_encoding is not None:
            output_dtype = torch_dtype_to_numpy(first_output_encoding[QCOM_DTYPE])
    output_raws = collect_outputs(output_data_folder, dtype=output_dtype)
    if output_encodings:
        output_raws = [
            dequantize_array(output, encoding)
            for output, encoding in zip(output_raws, output_encodings)
        ]
    if not output_raws:
        raise RuntimeError(
            f"No device outputs were pulled for {workspace_name}. Check device-side QNN runner logs."
        )
    return output_raws


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "--arch",
        help="Model architecture to export.",
        choices=("llama2", "gpt2"),
        default="llama2",
        type=str,
    )

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        default="./llama2",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint to load before export.",
        default=None,
    )

    parser.add_argument(
        "--params",
        help="Optional params json file. Defaults depend on --arch.",
        default=None,
    )

    parser.add_argument(
        "--quant",
        help="Quantization mode for export.",
        choices=("none", "8a8w"),
        default="none",
        type=str,
    )

    parser.add_argument(
        "--calib_steps",
        help="Number of repeated calibration batches for quantized export.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--quantizer_recipe",
        help="Quantizer recipe to use for quantized export.",
        choices=("generic", "llm"),
        default="generic",
        type=str,
    )

    parser.add_argument(
        "--pipeline",
        help="Export and execution pipeline.",
        choices=("monolithic", "split"),
        default="monolithic",
        type=str,
    )

    parser.add_argument(
        "--enable_mha2sha",
        help="Enable the experimental QNN multi-head-attention to single-head-attention lowering pass.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--enable_linear_to_conv2d",
        help="Enable experimental linear-to-conv2d conversion during QNN lowering.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--torch_num_threads",
        help="Maximum number of host-side PyTorch threads to use during export.",
        default=1,
        type=int,
    )

    args = parser.parse_args()

    if args.arch == "gpt2" and args.artifact == "./llama2":
        args.artifact = "./gpt2"

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    if args.arch == "llama2":
        from models.llama2 import (
            ModelArgs,
            Llama2DecoderModel,
            Llama2EmbeddingModel,
            Llama2Model,
        )

        params_path = args.params or "models/llama2_1b_config.json"
        with open(params_path, "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=128,
            max_batch_size=1,
            **params,
        )
        model = Llama2Model(model_args)
        sample_inputs = (torch.ones(2, 128, dtype=torch.int32),)
        model_prefix = "llama2"
        embedding_model = Llama2EmbeddingModel(model.model).eval()
        decoder_model = Llama2DecoderModel(model.model, model.lm_head).eval()
    else:
        from models.gpt2 import (
            GPT2Config,
            GPT2DecoderModel,
            GPT2EmbeddingModel,
            GPT2Model,
        )

        params_path = args.params or "models/gpt2_demo_config.json"
        with open(params_path, "r") as f:
            params = json.loads(f.read())

        model_args = GPT2Config(**params)
        model = GPT2Model(model_args)
        sample_inputs = (
            torch.ones(2, model_args.block_size, dtype=torch.int32),
        )
        model_prefix = "gpt2"
        embedding_model = GPT2EmbeddingModel(model.transformer).eval()
        decoder_model = GPT2DecoderModel(model.transformer, model.lm_head).eval()

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    quant_dtype = None
    custom_quantizer = None
    pte_filename = f"{model_prefix}_qnn"

    if args.quant == "8a8w":
        quant_dtype = QuantDtype.use_8a8w
        if args.quantizer_recipe == "llm":
            custom_quantizer = make_llm_quantizer(quant_dtype)
        else:
            custom_quantizer = make_quantizer(quant_dtype=quant_dtype)
        pte_filename = f"{model_prefix}_qnn_8a8w"
    use_quantized_io = quant_dtype is not None and not args.online_prepare

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    if args.pipeline == "split":
        with torch.no_grad():
            decoder_example_input = embedding_model(sample_inputs[0]).detach()

        quant_suffix = "_8a8w" if quant_dtype is not None else ""
        embedding_pte_filename = f"{model_prefix}_embedding_qnn{quant_suffix}"
        decoder_pte_filename = f"{model_prefix}_decoder_qnn{quant_suffix}"
        embedding_artifact = f"{args.artifact}/embedding"
        decoder_artifact = f"{args.artifact}/decoder"
        os.makedirs(embedding_artifact, exist_ok=True)
        os.makedirs(decoder_artifact, exist_ok=True)

        embedding_calibration_dataset = None
        decoder_calibration_dataset = None
        if quant_dtype is not None:
            embedding_calibration_dataset = [
                (sample_inputs[0],) for _ in range(args.calib_steps)
            ]
            decoder_calibration_dataset = [
                (decoder_example_input,) for _ in range(args.calib_steps)
            ]

        embedding_quant_io_info = build_executorch_binary(
            embedding_model,
            sample_inputs,
            args.model,
            f"{embedding_artifact}/{embedding_pte_filename}",
            dataset=embedding_calibration_dataset,
            quant_dtype=quant_dtype,
            custom_quantizer=custom_quantizer,
            shared_buffer=args.shared_buffer,
            online_prepare=args.online_prepare,
            use_mha2sha=False,
            convert_linear_to_conv2d=args.enable_linear_to_conv2d,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            quantized_io=use_quantized_io,
        )
        decoder_quant_io_info = build_executorch_binary(
            decoder_model,
            (decoder_example_input,),
            args.model,
            f"{decoder_artifact}/{decoder_pte_filename}",
            dataset=decoder_calibration_dataset,
            quant_dtype=quant_dtype,
            custom_quantizer=custom_quantizer,
            shared_buffer=args.shared_buffer,
            online_prepare=args.online_prepare,
            use_mha2sha=args.enable_mha2sha,
            convert_linear_to_conv2d=args.enable_linear_to_conv2d,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            quantized_io=use_quantized_io,
        )
    else:
        calibration_dataset = None
        if quant_dtype is not None:
            calibration_dataset = [(sample_inputs[0],) for _ in range(args.calib_steps)]

        quant_io_info = build_executorch_binary(
            model.eval(),
            sample_inputs,
            args.model,
            f"{args.artifact}/{pte_filename}",
            dataset=calibration_dataset,
            # custom_annotations=(),
            quant_dtype=quant_dtype,
            custom_quantizer=custom_quantizer,
            shared_buffer=args.shared_buffer,
            online_prepare=args.online_prepare,
            use_mha2sha=args.enable_mha2sha,
            convert_linear_to_conv2d=args.enable_linear_to_conv2d,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            quantized_io=use_quantized_io,
        )

    if args.compile_only:
        sys.exit(0)

    if args.pipeline == "split":
        embedding_output_raws = run_pte_on_device(
            args,
            f"{args.artifact}/embedding/{embedding_pte_filename}.pte",
            f"{args.artifact}/embedding_run",
            sample_inputs,
            embedding_pte_filename,
            input_encodings=(
                embedding_quant_io_info["input_encodings"]
                if embedding_quant_io_info
                else None
            ),
            output_encodings=(
                embedding_quant_io_info["output_encodings"]
                if embedding_quant_io_info
                else None
            ),
        )
        decoder_input = torch.from_numpy(embedding_output_raws[0]).reshape(
            decoder_example_input.shape
        )
        output_raws = run_pte_on_device(
            args,
            f"{args.artifact}/decoder/{decoder_pte_filename}.pte",
            f"{args.artifact}/decoder_run",
            (decoder_input,),
            decoder_pte_filename,
            input_encodings=(
                decoder_quant_io_info["input_encodings"]
                if decoder_quant_io_info
                else None
            ),
            output_encodings=(
                decoder_quant_io_info["output_encodings"]
                if decoder_quant_io_info
                else None
            ),
        )
    else:
        output_raws = run_pte_on_device(
            args,
            f"{args.artifact}/{pte_filename}.pte",
            args.artifact,
            sample_inputs,
            pte_filename,
            input_encodings=(
                quant_io_info["input_encodings"] if quant_io_info else None
            ),
            output_encodings=(
                quant_io_info["output_encodings"] if quant_io_info else None
            ),
        )

    x86_golden = model.eval()(sample_inputs[0])
    device_output = torch.from_numpy(output_raws[0]).reshape(x86_golden.size())
    result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-2)).tolist()

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "is_close": result,
                    }
                )
            )
    else:
        print(f"is_close? {result}")
        print(f"x86_golden {x86_golden}")
        print(f"device_out {device_output}")
