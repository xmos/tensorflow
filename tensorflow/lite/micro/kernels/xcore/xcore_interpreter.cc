// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"

namespace tflite {
namespace micro {
namespace xcore {

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   tflite::MicroAllocator* allocator,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   tflite::MicroProfiler* profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, profiler),
      dispatcher_(reporter, use_current_thread) {
  SetDispatcher(&dispatcher_);
}

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   tflite::MicroAllocator* allocator,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   XCoreProfiler* profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, profiler),
      dispatcher_(reporter, use_current_thread) {
  SetDispatcher(&dispatcher_);
  if (profiler) {
    profiler->Init(allocator, operators_size());
  }
}

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   uint8_t* arena, size_t arena_size,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   XCoreProfiler* profiler)
    : XCoreInterpreter::XCoreInterpreter(
          model, resolver, MicroAllocator::Create(arena, arena_size, reporter),
          reporter, use_current_thread, profiler) {}

TfLiteTensor* XCoreInterpreter::tensor(size_t tensor_index) {
  auto ctx = context();
  return ctx.GetTensor(&ctx, tensor_index);
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
