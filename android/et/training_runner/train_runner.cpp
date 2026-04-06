#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/serialize/serialize.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::FileDataLoader;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::extension::training::TrainingModule;
using executorch::extension::training::optimizer::SGD;
using executorch::extension::training::optimizer::SGDOptions;
using executorch::runtime::Error;

struct Options {
  std::string model_path;
  std::string ptd_path;
  std::string save_ptd_path;
  std::string train_tokens_path;
  std::string eval_tokens_path;
  std::string method_name = "forward";
  int steps = 50;
  int eval_steps = 8;
  int batch_size = 8;
  int block_size = 16;
  int vocab_size = 32;
  int dataset_stride = 0;
  int warmup_steps = 0;
  int log_every = 10;
  int seed = 0;
  bool tie_embedding_head = false;
  double learning_rate = 0.05;
};

struct BatchBuffers {
  std::vector<int64_t> tokens;
  std::vector<int64_t> labels;
};

struct TokenDataset {
  std::vector<int32_t> tokens;
};

struct LoopStats {
  double average_loss = std::numeric_limits<double>::quiet_NaN();
  double average_accuracy = std::numeric_limits<double>::quiet_NaN();
  double average_data_prep_ms = 0.0;
  double average_step_ms = 0.0;
  double last_loss = std::numeric_limits<double>::quiet_NaN();
};

void print_usage() {
  std::cout
      << "Usage: gpt2_training_runner --model_path <path> [options]\n"
      << "Options:\n"
      << "  --ptd_path <path>         External weights file\n"
      << "  --save_ptd_path <path>    Save trained weights after the run\n"
      << "  --train_tokens_path <path> Pretokenized int32 train tokens\n"
      << "  --eval_tokens_path <path> Pretokenized int32 eval tokens\n"
      << "  --method_name <name>      ExecuTorch method name (default: forward)\n"
      << "  --steps <int>             Training steps (default: 50)\n"
      << "  --eval_steps <int>        Evaluation steps before/after training (default: 8)\n"
      << "  --batch_size <int>        Batch size (default: 8)\n"
      << "  --block_size <int>        Sequence length (default: 16)\n"
      << "  --vocab_size <int>        Synthetic vocab size (default: 32)\n"
      << "  --dataset_stride <int>    Sliding-window stride for token datasets (default: block_size)\n"
      << "  --learning_rate <float>   SGD learning rate (default: 0.05)\n"
      << "  --tie_embedding_head      Sum embedding/lm_head grads and re-tie after SGD step\n"
      << "  --warmup_steps <int>      Unmeasured warmup steps (default: 0)\n"
      << "  --log_every <int>         Logging interval in steps (default: 10)\n"
      << "  --seed <int>              Reserved seed knob for repeatability (default: 0)\n";
}

bool parse_int(const std::string& value, int* out) {
  try {
    *out = std::stoi(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_double(const std::string& value, double* out) {
  try {
    *out = std::stod(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool consume_arg(
    int argc,
    char** argv,
    int* index,
    const std::string& flag,
    std::string* value) {
  const std::string arg(argv[*index]);
  if (arg == flag) {
    if (*index + 1 >= argc) {
      return false;
    }
    *value = argv[++(*index)];
    return true;
  }
  const std::string prefix = flag + "=";
  if (arg.rfind(prefix, 0) == 0) {
    *value = arg.substr(prefix.size());
    return true;
  }
  return false;
}

bool parse_args(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return false;
    }

    std::string value;
    if (consume_arg(argc, argv, &i, "--model_path", &value)) {
      options->model_path = value;
    } else if (consume_arg(argc, argv, &i, "--ptd_path", &value)) {
      options->ptd_path = value;
    } else if (consume_arg(argc, argv, &i, "--save_ptd_path", &value)) {
      options->save_ptd_path = value;
    } else if (consume_arg(argc, argv, &i, "--train_tokens_path", &value)) {
      options->train_tokens_path = value;
    } else if (consume_arg(argc, argv, &i, "--eval_tokens_path", &value)) {
      options->eval_tokens_path = value;
    } else if (consume_arg(argc, argv, &i, "--method_name", &value)) {
      options->method_name = value;
    } else if (
        consume_arg(argc, argv, &i, "--steps", &value) &&
        parse_int(value, &options->steps)) {
    } else if (
        consume_arg(argc, argv, &i, "--eval_steps", &value) &&
        parse_int(value, &options->eval_steps)) {
    } else if (
        consume_arg(argc, argv, &i, "--batch_size", &value) &&
        parse_int(value, &options->batch_size)) {
    } else if (
        consume_arg(argc, argv, &i, "--block_size", &value) &&
        parse_int(value, &options->block_size)) {
    } else if (
        consume_arg(argc, argv, &i, "--vocab_size", &value) &&
        parse_int(value, &options->vocab_size)) {
    } else if (
        consume_arg(argc, argv, &i, "--dataset_stride", &value) &&
        parse_int(value, &options->dataset_stride)) {
    } else if (
        consume_arg(argc, argv, &i, "--warmup_steps", &value) &&
        parse_int(value, &options->warmup_steps)) {
    } else if (
        consume_arg(argc, argv, &i, "--log_every", &value) &&
        parse_int(value, &options->log_every)) {
    } else if (
        consume_arg(argc, argv, &i, "--seed", &value) &&
        parse_int(value, &options->seed)) {
    } else if (
        consume_arg(argc, argv, &i, "--learning_rate", &value) &&
        parse_double(value, &options->learning_rate)) {
    } else if (arg == "--tie_embedding_head") {
      options->tie_embedding_head = true;
    } else {
      std::cerr << "Unknown or malformed argument: " << arg << "\n";
      print_usage();
      return false;
    }
  }

  if (options->model_path.empty()) {
    std::cerr << "--model_path is required\n";
    print_usage();
    return false;
  }
  if (options->steps < 0 || options->eval_steps < 0 || options->warmup_steps < 0) {
    std::cerr << "steps/eval_steps/warmup_steps must be non-negative\n";
    return false;
  }
  if (options->batch_size <= 0 || options->block_size <= 0 || options->vocab_size <= 0) {
    std::cerr << "batch_size/block_size/vocab_size must be positive\n";
    return false;
  }
  if (options->learning_rate <= 0.0) {
    std::cerr << "learning_rate must be positive\n";
    return false;
  }
  options->log_every = std::max(1, options->log_every);
  return true;
}

long read_proc_status_kb(const std::string& key) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(key + ":", 0) != 0) {
      continue;
    }
    std::istringstream iss(line);
    std::string ignored_key;
    long value_kb = -1;
    std::string unit;
    iss >> ignored_key >> value_kb >> unit;
    return value_kb;
  }
  return -1;
}

void fill_batch(BatchBuffers* buffers, const Options& options, int offset) {
  const size_t numel =
      static_cast<size_t>(options.batch_size) * options.block_size;
  buffers->tokens.resize(numel);
  buffers->labels.resize(numel);

  for (int batch_idx = 0; batch_idx < options.batch_size; ++batch_idx) {
    for (int pos = 0; pos < options.block_size; ++pos) {
      const size_t index =
          static_cast<size_t>(batch_idx) * options.block_size + pos;
      const int64_t token =
          static_cast<int64_t>((batch_idx + offset + pos) % options.vocab_size);
      buffers->tokens[index] = token;
      buffers->labels[index] = token;
    }
  }
}

TokenDataset load_token_dataset(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    ET_LOG(Error, "Failed to open token dataset file: %s", path.c_str());
    std::exit(1);
  }

  file.seekg(0, std::ios::end);
  const auto num_bytes = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  if (num_bytes % sizeof(int32_t) != 0) {
    ET_LOG(
        Error,
        "Token dataset size must be a multiple of %zu bytes, got %zu",
        sizeof(int32_t),
        num_bytes);
    std::exit(1);
  }

  TokenDataset dataset;
  dataset.tokens.resize(num_bytes / sizeof(int32_t));
  file.read(
      reinterpret_cast<char*>(dataset.tokens.data()),
      static_cast<std::streamsize>(num_bytes));
  if (!file) {
    ET_LOG(Error, "Failed to read token dataset file: %s", path.c_str());
    std::exit(1);
  }
  return dataset;
}

void fill_batch_from_dataset(
    BatchBuffers* buffers,
    const Options& options,
    const TokenDataset& dataset,
    int offset) {
  if (dataset.tokens.size() < static_cast<size_t>(options.block_size)) {
    ET_LOG(
        Error,
        "Token dataset is too short: %zu < block_size %d",
        dataset.tokens.size(),
        options.block_size);
    std::exit(1);
  }

  const size_t numel =
      static_cast<size_t>(options.batch_size) * options.block_size;
  buffers->tokens.resize(numel);
  buffers->labels.resize(numel);

  const int stride =
      options.dataset_stride > 0 ? options.dataset_stride : options.block_size;
  const size_t max_start =
      dataset.tokens.size() - static_cast<size_t>(options.block_size);
  const size_t num_windows = max_start / static_cast<size_t>(stride) + 1;

  for (int batch_idx = 0; batch_idx < options.batch_size; ++batch_idx) {
    const size_t window_index =
        (static_cast<size_t>(offset) * options.batch_size + batch_idx) %
        num_windows;
    const size_t start = window_index * static_cast<size_t>(stride);
    for (int pos = 0; pos < options.block_size; ++pos) {
      const size_t index =
          static_cast<size_t>(batch_idx) * options.block_size + pos;
      const int64_t token =
          static_cast<int64_t>(dataset.tokens[start + static_cast<size_t>(pos)]);
      buffers->tokens[index] = token;
      buffers->labels[index] = token;
    }
  }
}

std::pair<TensorPtr, TensorPtr> make_batch_tensors(
    BatchBuffers* buffers,
    const Options& options) {
  std::vector<executorch::aten::SizesType> shape = {
      options.batch_size,
      options.block_size,
  };
  auto tokens = from_blob(
      buffers->tokens.data(),
      shape,
      ScalarType::Long,
      executorch::aten::TensorShapeDynamism::STATIC);
  auto labels = from_blob(
      buffers->labels.data(),
      shape,
      ScalarType::Long,
      executorch::aten::TensorShapeDynamism::STATIC);
  return {tokens, labels};
}

double tensor_to_double(const Tensor& tensor) {
  return static_cast<double>(tensor.const_data_ptr<float>()[0]);
}

bool has_suffix(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
      value.substr(value.size() - suffix.size()) == suffix;
}

const Tensor* find_named_tensor(
    const std::map<std::string_view, Tensor>& tensor_map,
    std::initializer_list<std::string_view> candidate_suffixes) {
  for (const auto& [name, tensor] : tensor_map) {
    for (const auto& suffix : candidate_suffixes) {
      if (name == suffix || has_suffix(name, suffix)) {
        return &tensor;
      }
    }
  }
  return nullptr;
}

void sum_tied_embedding_head_gradients(
    TrainingModule* module,
    const Options& options) {
  if (!options.tie_embedding_head) {
    return;
  }

  auto grad_res = module->named_gradients(options.method_name);
  if (!grad_res.ok()) {
    ET_LOG(
        Error,
        "named_gradients failed during tie_embedding_head with error 0x%" PRIx32,
        static_cast<uint32_t>(grad_res.error()));
    std::exit(1);
  }

  const Tensor* embedding_grad = find_named_tensor(
      grad_res.get(),
      {"model.transformer.wte.weight", "transformer.wte.weight"});
  const Tensor* lm_head_grad = find_named_tensor(
      grad_res.get(),
      {"model.lm_head.weight", "lm_head.weight"});
  if (embedding_grad == nullptr || lm_head_grad == nullptr) {
    ET_LOG(
        Error,
        "tie_embedding_head requested but embedding/lm_head gradients were not found");
    std::exit(1);
  }
  if (embedding_grad->numel() != lm_head_grad->numel()) {
    ET_LOG(
        Error,
        "tie_embedding_head gradient size mismatch: %zu vs %zu",
        embedding_grad->numel(),
        lm_head_grad->numel());
    std::exit(1);
  }

  auto emb_grad = *embedding_grad;
  auto head_grad = *lm_head_grad;
  float* emb_ptr = emb_grad.mutable_data_ptr<float>();
  float* head_ptr = head_grad.mutable_data_ptr<float>();
  for (size_t idx = 0; idx < emb_grad.numel(); ++idx) {
    emb_ptr[idx] += head_ptr[idx];
    head_ptr[idx] = emb_ptr[idx];
  }
}

void retie_embedding_head_parameters(
    TrainingModule* module,
    const Options& options) {
  if (!options.tie_embedding_head) {
    return;
  }

  auto param_res = module->named_parameters(options.method_name);
  if (!param_res.ok()) {
    ET_LOG(
        Error,
        "named_parameters failed during tie_embedding_head with error 0x%" PRIx32,
        static_cast<uint32_t>(param_res.error()));
    std::exit(1);
  }

  const Tensor* embedding_param = find_named_tensor(
      param_res.get(),
      {"model.transformer.wte.weight", "transformer.wte.weight"});
  const Tensor* lm_head_param = find_named_tensor(
      param_res.get(),
      {"model.lm_head.weight", "lm_head.weight"});
  if (embedding_param == nullptr || lm_head_param == nullptr) {
    ET_LOG(
        Error,
        "tie_embedding_head requested but embedding/lm_head parameters were not found");
    std::exit(1);
  }
  if (embedding_param->numel() != lm_head_param->numel()) {
    ET_LOG(
        Error,
        "tie_embedding_head parameter size mismatch: %zu vs %zu",
        embedding_param->numel(),
        lm_head_param->numel());
    std::exit(1);
  }

  auto emb_param = *embedding_param;
  auto head_param = *lm_head_param;
  const float* emb_ptr = emb_param.const_data_ptr<float>();
  float* head_ptr = head_param.mutable_data_ptr<float>();
  std::copy(emb_ptr, emb_ptr + emb_param.numel(), head_ptr);
}

double token_accuracy(
    const Tensor& predictions,
    const std::vector<int64_t>& labels,
    int batch_size,
    int block_size) {
  const auto* predicted = predictions.const_data_ptr<int64_t>();
  size_t correct = 0;
  size_t total = 0;
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int pos = 0; pos + 1 < block_size; ++pos) {
      const size_t index =
          static_cast<size_t>(batch_idx) * block_size + pos;
      if (predicted[index] == labels[index + 1]) {
        ++correct;
      }
      ++total;
    }
  }
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(correct) / total;
}

void fill_batch_for_step(
    BatchBuffers* buffers,
    const Options& options,
    const TokenDataset* dataset,
    int offset) {
  if (dataset != nullptr) {
    fill_batch_from_dataset(buffers, options, *dataset, offset);
    return;
  }
  fill_batch(buffers, options, offset);
}

LoopStats run_loop(
    TrainingModule* module,
    SGD* optimizer,
    const Options& options,
    int num_steps,
    int start_offset,
    bool log_progress,
    const TokenDataset* dataset) {
  LoopStats stats;
  if (num_steps == 0) {
    return stats;
  }

  BatchBuffers buffers;
  double total_loss = 0.0;
  double total_accuracy = 0.0;
  double total_data_prep_ms = 0.0;
  double total_step_ms = 0.0;

  for (int step = 0; step < num_steps; ++step) {
    const auto step_begin = std::chrono::steady_clock::now();
    fill_batch_for_step(&buffers, options, dataset, start_offset + step);
    auto tensors = make_batch_tensors(&buffers, options);
    const auto data_ready = std::chrono::steady_clock::now();

    auto result = module->execute_forward_backward(
        options.method_name, {*tensors.first, *tensors.second});
    if (!result.ok()) {
      ET_LOG(
          Error,
          "execute_forward_backward failed at step %d with error 0x%" PRIx32,
          step,
          static_cast<uint32_t>(result.error()));
      std::exit(1);
    }
    if (optimizer != nullptr) {
      sum_tied_embedding_head_gradients(module, options);
      auto opt_err = optimizer->step(module->named_gradients(options.method_name).get());
      if (opt_err != Error::Ok) {
        ET_LOG(
            Error,
            "optimizer.step failed at step %d with error 0x%" PRIx32,
            step,
            static_cast<uint32_t>(opt_err));
        std::exit(1);
      }
      retie_embedding_head_parameters(module, options);
    }

    const auto step_end = std::chrono::steady_clock::now();
    const Tensor& loss_tensor = result.get()[0].toTensor();
    const double loss = tensor_to_double(loss_tensor);
    double accuracy = std::numeric_limits<double>::quiet_NaN();
    if (result.get().size() > 1) {
      const Tensor& predictions = result.get()[1].toTensor();
      accuracy = token_accuracy(
          predictions, buffers.labels, options.batch_size, options.block_size);
    }

    total_loss += loss;
    total_accuracy += accuracy;
    stats.last_loss = loss;
    total_data_prep_ms += std::chrono::duration<double, std::milli>(
                              data_ready - step_begin)
                              .count();
    total_step_ms +=
        std::chrono::duration<double, std::milli>(step_end - step_begin).count();

    if (log_progress && ((step + 1) % options.log_every == 0 || step == num_steps - 1)) {
      ET_LOG(
          Info,
          "step=%d loss=%.6f token_acc=%.4f data_prep_ms=%.3f step_ms=%.3f rss_kb=%ld",
          step + 1,
          loss,
          accuracy,
          std::chrono::duration<double, std::milli>(data_ready - step_begin).count(),
          std::chrono::duration<double, std::milli>(step_end - step_begin).count(),
          read_proc_status_kb("VmRSS"));
    }
  }

  stats.average_loss = total_loss / num_steps;
  stats.average_accuracy = total_accuracy / num_steps;
  stats.average_data_prep_ms = total_data_prep_ms / num_steps;
  stats.average_step_ms = total_step_ms / num_steps;
  return stats;
}

void maybe_save_ptd(
    TrainingModule* module,
    const Options& options) {
  if (options.save_ptd_path.empty()) {
    return;
  }
  auto param_res = module->named_parameters(options.method_name);
  if (!param_res.ok()) {
    ET_LOG(
        Error,
        "named_parameters failed during save with error 0x%" PRIx32,
        static_cast<uint32_t>(param_res.error()));
    std::exit(1);
  }

  std::map<std::string, Tensor> param_map;
  for (const auto& [name, tensor] : param_res.get()) {
    param_map.emplace(std::string(name), tensor);
  }
  auto save_err = executorch::extension::flat_tensor::save_ptd(
      options.save_ptd_path, param_map, 16);
  if (save_err != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to save trained weights to %s with error 0x%" PRIx32,
        options.save_ptd_path.c_str(),
        static_cast<uint32_t>(save_err));
    std::exit(1);
  }
}

void print_result(const std::string& key, double value) {
  std::cout << std::fixed << std::setprecision(6) << "RESULT " << key << "="
            << value << "\n";
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  Options options;
  if (!parse_args(argc, argv, &options)) {
    return options.model_path.empty() ? 1 : 0;
  }

  ET_LOG(
      Info,
      "Loading training module model=%s ptd=%s method=%s train_tokens=%s eval_tokens=%s",
      options.model_path.c_str(),
      options.ptd_path.empty() ? "<none>" : options.ptd_path.c_str(),
      options.method_name.c_str(),
      options.train_tokens_path.empty() ? "<synthetic>" : options.train_tokens_path.c_str(),
      options.eval_tokens_path.empty() ? "<train/default>" : options.eval_tokens_path.c_str());

  auto loader_res = FileDataLoader::from(options.model_path.c_str());
  if (!loader_res.ok()) {
    ET_LOG(Error, "Failed to open model file: %s", options.model_path.c_str());
    return 1;
  }
  auto loader =
      std::make_unique<FileDataLoader>(std::move(loader_res.get()));

  std::unique_ptr<FileDataLoader> ptd_loader = nullptr;
  if (!options.ptd_path.empty()) {
    auto ptd_loader_res = FileDataLoader::from(options.ptd_path.c_str());
    if (!ptd_loader_res.ok()) {
      ET_LOG(Error, "Failed to open ptd file: %s", options.ptd_path.c_str());
      return 1;
    }
    ptd_loader =
        std::make_unique<FileDataLoader>(std::move(ptd_loader_res.get()));
  }

  TrainingModule module(
      std::move(loader), nullptr, nullptr, nullptr, std::move(ptd_loader));

  std::unique_ptr<TokenDataset> train_dataset = nullptr;
  if (!options.train_tokens_path.empty()) {
    train_dataset =
        std::make_unique<TokenDataset>(load_token_dataset(options.train_tokens_path));
  }
  std::unique_ptr<TokenDataset> eval_dataset = nullptr;
  if (!options.eval_tokens_path.empty()) {
    eval_dataset =
        std::make_unique<TokenDataset>(load_token_dataset(options.eval_tokens_path));
  }
  const TokenDataset* effective_eval_dataset =
      eval_dataset != nullptr ? eval_dataset.get() : train_dataset.get();

  auto param_res = module.named_parameters(options.method_name);
  if (!param_res.ok()) {
    ET_LOG(
        Error,
        "named_parameters failed with error 0x%" PRIx32,
        static_cast<uint32_t>(param_res.error()));
    return 1;
  }
  SGD optimizer(param_res.get(), SGDOptions(options.learning_rate));

  for (int warmup_step = 0; warmup_step < options.warmup_steps; ++warmup_step) {
    (void)run_loop(
        &module,
        nullptr,
        options,
        1,
        warmup_step,
        /*log_progress=*/false,
        train_dataset.get());
  }

  const auto wall_begin = std::chrono::steady_clock::now();
  const LoopStats pre_eval = run_loop(
      &module,
      nullptr,
      options,
      options.eval_steps,
      0,
      /*log_progress=*/false,
      effective_eval_dataset);
  const LoopStats train_stats = run_loop(
      &module,
      &optimizer,
      options,
      options.steps,
      0,
      /*log_progress=*/true,
      train_dataset.get());
  const LoopStats post_eval = run_loop(
      &module,
      nullptr,
      options,
      options.eval_steps,
      0,
      /*log_progress=*/false,
      effective_eval_dataset);
  const auto wall_end = std::chrono::steady_clock::now();

  maybe_save_ptd(&module, options);

  print_result("pre_eval_loss", pre_eval.average_loss);
  print_result("pre_eval_token_acc", pre_eval.average_accuracy);
  print_result("post_eval_loss", post_eval.average_loss);
  print_result("post_eval_token_acc", post_eval.average_accuracy);
  print_result("last_train_loss", train_stats.last_loss);
  print_result("train_avg_loss", train_stats.average_loss);
  print_result("train_avg_token_acc", train_stats.average_accuracy);
  print_result("avg_data_prep_ms", train_stats.average_data_prep_ms);
  print_result("avg_step_ms", train_stats.average_step_ms);
  print_result(
      "wall_ms",
      std::chrono::duration<double, std::milli>(wall_end - wall_begin).count());
  print_result("vmrss_kb", static_cast<double>(read_proc_status_kb("VmRSS")));
  print_result("vmhwm_kb", static_cast<double>(read_proc_status_kb("VmHWM")));
  return 0;
}
