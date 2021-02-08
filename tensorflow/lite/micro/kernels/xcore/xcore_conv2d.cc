#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

struct Conv2DOpData {
  Conv2DParams params;
  ExecutionPlan execution_plan;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct Conv2DThreadData {
  // TODO: none of these belong to this struct
  nn_image_t *Y;
  const nn_image_t *X;
  const nn_tensor_t *K;
  const nn_bso_block_t *BSO;
  const nn_image_params_t *x_image;
  const nn_image_params_t *y_image;
  const nn_window_params_t *window;   // not used by 1x1
  int8_t zero_point;                  // not used by 1x1
  nn_conv2d_depthwise_flags_e flags;  // only for depthwise

  nn_window_op_job_params_t job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_shallow_thread_worker(void *context) {
  Conv2DThreadData *td = (Conv2DThreadData *)context;
  conv2d_shallowin_ext(td->Y, td->X, td->K, td->BSO, td->zero_point,
                       td->x_image, td->y_image, td->window, &td->job,
                       CONV2D_SHALLOWIN_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_deep_thread_worker(void *context) {
  Conv2DThreadData *td = (Conv2DThreadData *)context;
  conv2d_deep_ext(td->Y, td->X, td->K, td->BSO, td->zero_point, td->x_image,
                  td->y_image, td->window, &td->job, CONV2D_DEEP_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_1x1_thread_worker(void *context) {
  Conv2DThreadData *td = (Conv2DThreadData *)context;

  // TODO: consider changing the kernel to unify this job struct
  nn_conv2d_1x1_job_params_t job;
  job.start = td->job.start;
  job.size.channels = td->job.size.channels;
  job.size.pixels = td->job.size.rows * td->job.size.cols;
  conv2d_1x1_ext(td->Y, td->X, td->K, td->BSO, td->x_image, td->y_image, &job,
                 CONV2D_1X1_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_depthwise_thread_worker(void *context) {
  Conv2DThreadData *td = (Conv2DThreadData *)context;
  conv2d_depthwise_ext(td->Y, td->X, td->K, td->BSO, td->zero_point,
                       td->x_image, td->y_image, td->window, &td->job,
                       td->flags);
}
}

// -------------------------------------------------------------------- //
// kernel types
// -------------------------------------------------------------------- //

enum class Conv2DKernelType {
  DEEP,
  SHALLOW,
  ONE_BY_ONE,
  DEPTHWISE,
};

template <Conv2DKernelType kernel_type>
struct Conv2DKernel {
  static inline const thread_function_t get_worker() {
    if (kernel_type == Conv2DKernelType::DEEP) {
      return conv2d_deep_thread_worker;
    } else if (kernel_type == Conv2DKernelType::SHALLOW) {
      return conv2d_shallow_thread_worker;
    } else if (kernel_type == Conv2DKernelType::ONE_BY_ONE) {
      return conv2d_1x1_thread_worker;
    } else if (kernel_type == Conv2DKernelType::DEPTHWISE) {
      return conv2d_depthwise_thread_worker;
    } else {
      UNSUPPORTED_KERNEL_TYPE(Conv2DKernelType);
    }
  };
  static inline void calculate_worker_stack_size(size_t &stack_size) {
    if (kernel_type == Conv2DKernelType::DEEP) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_deep_thread_worker);
    } else if (kernel_type == Conv2DKernelType::SHALLOW) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_shallow_thread_worker);
    } else if (kernel_type == Conv2DKernelType::ONE_BY_ONE) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_1x1_thread_worker);
    } else if (kernel_type == Conv2DKernelType::DEPTHWISE) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_depthwise_thread_worker);
    } else {
      UNSUPPORTED_KERNEL_TYPE(Conv2DKernelType);
    }
  };
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto *op_data = reinterpret_cast<Conv2DOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(Conv2DOpData)));
  op_data->stack_scratch_index = -1;
  op_data->stack_size = 0;
  op_data->weights_scratch_index = -1;
  op_data->bias_scratch_index = -1;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op_data->params,
                       &op_data->execution_plan);

  return op_data;
}

TfLiteStatus PrepareCommon(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  Conv2DOpData *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);

  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 1), op_data->weights_scratch_index));
  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 2), op_data->bias_scratch_index));

  return kTfLiteOk;
}

template <Conv2DKernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_STATUS(PrepareCommon(context, node));

  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);

  // allocate the stack for thread workers
  Conv2DKernel<kernel_type>::calculate_worker_stack_size(op_data->stack_size);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->execution_plan.regions.GetSize(),
      &op_data->stack_scratch_index));

  if (kernel_type != Conv2DKernelType::ONE_BY_ONE) {
    auto const *weights = GetInput(context, node, 1);
    if (kernel_type == Conv2DKernelType::SHALLOW) {
      op_data->params.K_h = weights->dims->data[1];
    } else if (kernel_type == Conv2DKernelType::DEPTHWISE) {
      auto const *weights = GetInput(context, node, 1);
      op_data->params.K_h = weights->dims->data[0];
      op_data->params.K_w = weights->dims->data[1];
    } else if (kernel_type == Conv2DKernelType::DEEP) {
      op_data->params.K_h = weights->dims->data[1];
      op_data->params.K_w = weights->dims->data[2];
    }
  }

  return kTfLiteOk;
}

//**************************************
//**************************************
//**************************************
// Shallow
//**************************************
//**************************************
//**************************************
namespace shallow {

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DOpData *op = reinterpret_cast<Conv2DOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_shallow_thread_worker, stack,
                              op->stack_size);

  // create thread data
  int n_th = op->execution_plan.GetNumThreads();
  Conv2DThreadData thread_data[n_th];

  // setup params common to all thread workers
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};
  nn_image_params_t out_image = {(uint32_t)output->dims->data[1],
                                 (uint32_t)output->dims->data[2],
                                 (uint32_t)weights->dims->data[0]};
  nn_window_params_t conv_window = {
      {(uint32_t)op->params.K_h, (uint32_t)op->params.K_w},
      {-op->params.pad.top, -op->params.pad.left},
      {op->params.stride_h, op->params.stride_w}};

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;
  size_t biases_src_offset = 0;
  size_t weights_src_offset = 0;
  size_t weights_fetch_size;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    weights_fetch_size = input->dims->data[3] * weights->dims->data[1] *
                         weights->dims->data[2] * changrp.size;
    dispatcher->FetchBuffer(&tK, &weights->data.int8[weights_src_offset],
                            weights_fetch_size);
    weights_src_offset += weights_fetch_size;
    dispatcher->FetchBuffer((int8_t **)&tBSO,
                            &bso->data.int8[biases_src_offset],
                            bso_changrp_bytes);
    biases_src_offset += bso_changrp_bytes;

    // create tasks
    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

      thread_data[i_rg].Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].K = (const nn_tensor_t *)tK;
      thread_data[i_rg].BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].zero_point = op->params.pad.zero_point;
      thread_data[i_rg].x_image = &in_image;
      thread_data[i_rg].y_image = &out_image;
      thread_data[i_rg].window = &conv_window;
      thread_data[i_rg].job = {{region.top, region.left, changrp.start},
                               {region.rows, region.cols, changrp.size}};
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace shallow

//**************************************
//**************************************
//**************************************
// Conv2D_Deep
//**************************************
//**************************************
//**************************************
namespace deep {

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DOpData *op = reinterpret_cast<Conv2DOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_deep_thread_worker, stack, op->stack_size);

  // create thread data
  int n_th = op->execution_plan.GetNumThreads();
  Conv2DThreadData thread_data[n_th];

  // setup params common to all thread workers
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};
  nn_image_params_t out_image = {(uint32_t)output->dims->data[1],
                                 (uint32_t)output->dims->data[2],
                                 (uint32_t)weights->dims->data[0]};
  nn_window_params_t conv_window = {
      {(uint32_t)op->params.K_h, (uint32_t)op->params.K_w},
      {-op->params.pad.top, -op->params.pad.left},
      {op->params.stride_h, op->params.stride_w}};

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;
  size_t weights_src_offset = 0;
  size_t weights_fetch_size;
  size_t biases_src_offset = 0;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  // create tasks
  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    weights_fetch_size =
        input->dims->data[3] * op->params.K_h * op->params.K_w * changrp.size;
    dispatcher->FetchBuffer(&tK, &weights->data.int8[weights_src_offset],
                            weights_fetch_size);
    weights_src_offset += weights_fetch_size;
    dispatcher->FetchBuffer((int8_t **)&tBSO,
                            &bso->data.int8[biases_src_offset],
                            bso_changrp_bytes);
    biases_src_offset += bso_changrp_bytes;

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

      thread_data[i_rg].Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].K = (const nn_tensor_t *)tK;
      thread_data[i_rg].BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].zero_point = op->params.pad.zero_point;
      thread_data[i_rg].x_image = &in_image;
      thread_data[i_rg].y_image = &out_image;
      thread_data[i_rg].window = &conv_window;
      thread_data[i_rg].job = {{region.top, region.left, changrp.start},
                               {region.rows, region.cols, changrp.size}};
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace deep

//**************************************
//**************************************
//**************************************
// 1x1
//**************************************
//**************************************
//**************************************
namespace n1x1 {

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DOpData *op = reinterpret_cast<Conv2DOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_1x1_thread_worker, stack, op->stack_size);

  // create thread data
  Conv2DThreadData thread_data[op->execution_plan.GetNumThreads()];

  // setup params common to all thread workers
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};
  nn_image_params_t out_image = {(uint32_t)output->dims->data[1],
                                 (uint32_t)output->dims->data[2],
                                 (uint32_t)output->dims->data[3]};

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;
  size_t weights_src_offset = 0;
  size_t weights_fetch_size;
  size_t biases_src_offset = 0;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    weights_fetch_size = input->dims->data[3] * changrp.size;
    dispatcher->FetchBuffer(&tK, &weights->data.int8[weights_src_offset],
                            weights_fetch_size);
    weights_src_offset += weights_fetch_size;
    dispatcher->FetchBuffer((int8_t **)&tBSO,
                            &bso->data.int8[biases_src_offset],
                            bso_changrp_bytes);
    biases_src_offset += bso_changrp_bytes;

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

      thread_data[i_rg].Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].K = (const nn_tensor_t *)tK;
      thread_data[i_rg].BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].x_image = &in_image;
      thread_data[i_rg].y_image = &out_image;
      thread_data[i_rg].job = {{region.top, region.left, changrp.start},
                               {region.rows, region.cols, changrp.size}};
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace n1x1

//**************************************
//**************************************
//**************************************
// depthwise
//**************************************
//**************************************
//**************************************

namespace depthwise {

static void fetch_depthwise_subtensor(int8_t *dest, const int8_t *weights,
                                      const unsigned K_h, const unsigned K_w,
                                      const unsigned X_c,
                                      const unsigned start_channel,
                                      const unsigned channel_count) {
  assert(start_channel % 16 == 0);
  assert(channel_count % 4 == 0);

  Dispatcher *dispatcher = GetDispatcher();

  weights =
      &(weights[start_channel]);  // Address of weights[0][0][start_channel]

  // Total of K_h * K_w blocks, for a total of K_h*K_w*channel_count bytes
  for (int k = 0; k < K_h * K_w; k++) {
    dispatcher->FetchBuffer(&dest, weights, channel_count);
    // memcpy(dest, weights, channel_count);
    dest = &(dest[channel_count]);
    weights = &(weights[X_c]);
  }
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DOpData *op = reinterpret_cast<Conv2DOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_depthwise_thread_worker, stack,
                              op->stack_size);

  // create thread data and tasks
  int n_th = op->execution_plan.GetNumThreads();
  Conv2DThreadData thread_data[n_th];

  // setup params common to all thread workers
  int32_t C_out = output->dims->data[3];
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};
  nn_image_params_t out_image = {(uint32_t)output->dims->data[1],
                                 (uint32_t)output->dims->data[2],
                                 (uint32_t)C_out};
  nn_window_params_t conv_window = {
      {(uint32_t)op->params.K_h, (uint32_t)op->params.K_w},
      {-op->params.pad.top, -op->params.pad.left},
      {op->params.stride_h, op->params.stride_w}};

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;
  size_t biases_src_offset = 0;
  nn_conv2d_depthwise_flags_e flags = (nn_conv2d_depthwise_flags_e)0;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    if (op->weights_scratch_index >= 0) {
      // fetch the weights
      fetch_depthwise_subtensor(tK, weights->data.int8, op->params.K_h,
                                op->params.K_w, C_out, changrp.start,
                                changrp.size);
      flags = CONV2D_DEPTHWISE_FLAG_SLICED_K;
    } else {
      // use entire tensor
      tK = weights->data.int8;
    }

    if (op->weights_scratch_index >= 0) {
      // fetch the biases
      dispatcher->FetchBuffer((int8_t **)&tBSO,
                              &bso->data.int8[biases_src_offset],
                              bso_changrp_bytes);
      biases_src_offset += bso_changrp_bytes;
      // dispatcher->FetchBiases(&tBSO, bso->data.i16,
      //                         op->execution_plan.GetBiasScratchSize(),
      //                         changrp);
    } else {
      // use entire tensor
      tBSO = bso->data.i16;
    }

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

      thread_data[i_rg].Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].K = (const nn_tensor_t *)tK;
      thread_data[i_rg].BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].zero_point = op->params.pad.zero_point;
      thread_data[i_rg].x_image = &in_image;
      thread_data[i_rg].y_image = &out_image;
      thread_data[i_rg].window = &conv_window;
      thread_data[i_rg].job = {{region.top, region.left, changrp.start},
                               {region.rows, region.cols, changrp.size}};
      thread_data[i_rg].flags = flags;
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace depthwise
}  // namespace conv

TfLiteRegistration *Register_Conv2D_Deep() {
  static TfLiteRegistration r = {conv::Init, nullptr,
                                 conv::Prepare<conv::Conv2DKernelType::DEEP>,
                                 conv::deep::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_Shallow() {
  static TfLiteRegistration r = {conv::Init, nullptr,
                                 conv::Prepare<conv::Conv2DKernelType::SHALLOW>,
                                 conv::shallow::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_1x1() {
  static TfLiteRegistration r = {
      conv::Init, nullptr, conv::Prepare<conv::Conv2DKernelType::ONE_BY_ONE>,
      conv::n1x1::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_Depthwise() {
  static TfLiteRegistration r = {
      conv::Init, nullptr, conv::Prepare<conv::Conv2DKernelType::DEPTHWISE>,
      conv::depthwise::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
