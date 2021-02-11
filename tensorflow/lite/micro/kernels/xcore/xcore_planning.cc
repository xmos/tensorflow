// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

//*****************************
//*****************************
//*****************************
// ExecutionPlan
//*****************************
//*****************************
//*****************************
ExecutionPlan::ExecutionPlan()
    : n_threads_(0), bias_scratch_offset_(0), bias_scratch_size_(0) {}

void ExecutionPlan::SetWeightsScratchSize(size_t size) {
  // NOTE: Weights assumes to start at scratch offset 0
  //        so we do not need to store it
  bias_scratch_offset_ = size;
}
size_t ExecutionPlan::GetWeightsScratchSize() { return bias_scratch_offset_; }
size_t ExecutionPlan::GetWeightsScratchOffset() { return 0; }

void ExecutionPlan::SetBiasScratchSize(size_t size) {
  bias_scratch_size_ = size;
}
size_t ExecutionPlan::GetBiasScratchSize() { return bias_scratch_size_; }
size_t ExecutionPlan::GetBiasScratchOffset() { return bias_scratch_offset_; }

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite