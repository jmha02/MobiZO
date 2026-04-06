cmake_minimum_required(VERSION 3.24)
project(gpt2_training_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED EXECUTORCH_ROOT)
  message(FATAL_ERROR "EXECUTORCH_ROOT must point to the ExecuTorch source tree.")
endif()

if(NOT DEFINED EXECUTORCH_ANDROID_BUILD_DIR)
  message(
    FATAL_ERROR
      "EXECUTORCH_ANDROID_BUILD_DIR must point to an Android ExecuTorch build directory."
  )
endif()

list(
  PREPEND
  CMAKE_PREFIX_PATH
  "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake/ExecuTorch"
  "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake"
)

set(absl_DIR "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake/absl")
set(executorch_DIR "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake/ExecuTorch")
set(tokenizers_DIR "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake/tokenizers")
set(re2_DIR "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake/re2")
set(pcre2_DIR "${EXECUTORCH_ANDROID_BUILD_DIR}/lib/cmake/pcre2")

find_package(absl CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(tokenizers CONFIG REQUIRED)
find_package(executorch CONFIG REQUIRED)

add_library(
  mobizo_extension_training
  STATIC
  "${EXECUTORCH_ROOT}/extension/training/module/training_module.cpp"
  "${EXECUTORCH_ROOT}/extension/training/optimizer/sgd.cpp"
)
target_include_directories(
  mobizo_extension_training
  PUBLIC
  "${EXECUTORCH_ROOT}/.."
  "${EXECUTORCH_ANDROID_BUILD_DIR}/include"
)
target_link_libraries(
  mobizo_extension_training
  PUBLIC
  executorch_core
  kernels_util_all_deps
  extension_data_loader
  extension_module_static
  extension_tensor
  extension_flat_tensor
)

add_library(
  mobizo_flat_tensor_serialize
  STATIC
  "${EXECUTORCH_ROOT}/extension/flat_tensor/serialize/serialize.cpp"
)
target_include_directories(
  mobizo_flat_tensor_serialize
  PUBLIC
  "${EXECUTORCH_ROOT}/.."
  "${EXECUTORCH_ANDROID_BUILD_DIR}/include"
  "${EXECUTORCH_ANDROID_BUILD_DIR}/extension/flat_tensor/include"
  "${EXECUTORCH_ROOT}/third-party/flatbuffers/include"
)
target_link_libraries(
  mobizo_flat_tensor_serialize
  PUBLIC
  executorch_core
)

add_executable(gpt2_training_runner train_runner.cpp)
target_include_directories(
  gpt2_training_runner
  PRIVATE
  "${EXECUTORCH_ROOT}/.."
  "${EXECUTORCH_ANDROID_BUILD_DIR}/include"
)
target_link_libraries(
  gpt2_training_runner
  PRIVATE
  executorch
  executorch_core
  portable_ops_lib
  extension_data_loader
  extension_flat_tensor
  extension_tensor
  extension_threadpool
  mobizo_flat_tensor_serialize
  mobizo_extension_training
)
