#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
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

struct Conv2DArguments {
  nn_image_t *Y;
  const nn_image_t *X;
  const nn_tensor_t *K;
  const nn_bso_block_t *BSO;

  nn_image_params_t x_image;
  nn_image_params_t y_image;

  nn_window_params_t window;                    // not used by 1x1
  int8_t zero_point;                            // not used by 1x1
  nn_conv2d_depthwise_flags_e depthwise_flags;  // only for depthwise
};

struct Conv2DOpData {
  Conv2DArguments args;
  Conv2DParams params;
  ExecutionPlan execution_plan;
  int stack_scratch_index = -1;
  size_t stack_size = 0;
  int weights_scratch_index = -1;
  int bias_scratch_index = -1;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct Conv2DThreadData {
  Conv2DArguments *args;
  nn_window_op_job_params_t job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_shallow_thread_worker(void *context) {
  auto *td = static_cast<Conv2DThreadData *>(context);
  auto *args = td->args;
  conv2d_shallowin_ext(args->Y, args->X, args->K, args->BSO, args->zero_point,
                       &args->x_image, &args->y_image, &args->window, &td->job,
                       CONV2D_SHALLOWIN_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_deep_thread_worker(void *context) {
  auto *td = static_cast<Conv2DThreadData *>(context);
  auto *args = td->args;

  conv2d_deep_ext(args->Y, args->X, args->K, args->BSO, args->zero_point,
                  &args->x_image, &args->y_image, &args->window, &td->job,
                  CONV2D_DEEP_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_1x1_thread_worker(void *context) {
  auto *td = static_cast<Conv2DThreadData *>(context);
  auto *args = td->args;

  // TODO: consider changing the kernel to unify this job struct
  nn_conv2d_1x1_job_params_t job;
  job.start = td->job.start;
  job.size.channels = td->job.size.channels;
  job.size.pixels = td->job.size.rows * td->job.size.cols;
  conv2d_1x1_ext(args->Y, args->X, args->K, args->BSO, &args->x_image,
                 &args->y_image, &job, CONV2D_1X1_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_depthwise_thread_worker(void *context) {
  auto *td = static_cast<Conv2DThreadData *>(context);
  auto *args = td->args;
  conv2d_depthwise_ext(args->Y, args->X, args->K, args->BSO, args->zero_point,
                       &args->x_image, &args->y_image, &args->window, &td->job,
                       args->depthwise_flags);
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
  auto *op_data = intialize_persistent_buffer<Conv2DOpData>(context);

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op_data->params,
                       &op_data->execution_plan);

  return op_data;
}

TfLiteStatus PrepareCommon(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);

  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 1), op_data->weights_scratch_index));
  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 2), op_data->bias_scratch_index));

  const auto &input_shape = GetTensorShape(GetInput(context, node, 0));
  op_data->args.x_image = {(uint32_t)input_shape.Dims(1),
                           (uint32_t)input_shape.Dims(2),
                           (uint32_t)input_shape.Dims(3)};

  const auto &output_shape = GetTensorShape(GetOutput(context, node, 0));
  op_data->args.y_image = {(uint32_t)output_shape.Dims(1),
                           (uint32_t)output_shape.Dims(2),
                           (uint32_t)output_shape.Dims(3)};

  // not used by 1x1
  op_data->args.window.start = {-op_data->params.pad.top,
                                -op_data->params.pad.left};
  op_data->args.window.stride = {op_data->params.stride_h,
                                 op_data->params.stride_w};
  op_data->args.zero_point = op_data->params.pad.zero_point;

  return kTfLiteOk;
}

template <Conv2DKernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_STATUS(PrepareCommon(context, node));

  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);

  // allocate the stack for thread workers
  Conv2DKernel<kernel_type>::calculate_worker_stack_size(op_data->stack_size);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->execution_plan.regions.size(),
      &op_data->stack_scratch_index));

  const auto &weight_shape = GetTensorShape(GetInput(context, node, 1));
  if (kernel_type == Conv2DKernelType::ONE_BY_ONE) {
    op_data->args.window.shape = {1, 1};
  } else if (kernel_type == Conv2DKernelType::SHALLOW) {
    op_data->args.window.shape.height = weight_shape.Dims(1);
    op_data->args.window.shape.width = op_data->params.K_w;
  } else if (kernel_type == Conv2DKernelType::DEPTHWISE) {
    op_data->args.window.shape.height = weight_shape.Dims(0);
    op_data->args.window.shape.width = weight_shape.Dims(1);
  } else if (kernel_type == Conv2DKernelType::DEEP) {
    op_data->args.window.shape.height = weight_shape.Dims(1);
    op_data->args.window.shape.width = weight_shape.Dims(2);
  }

  return kTfLiteOk;
}

TfLiteStatus EvalCommon(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);
  op_data->args.Y = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalOutput(context, node, 0));
  op_data->args.X = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalInput(context, node, 0));
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
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  const TfLiteEvalTensor *weights =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bso = tflite::micro::GetEvalInput(context, node, 2);

  const RuntimeShape weights_shape = tflite::micro::GetTensorShape(weights);

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
  for (int j{0}; j < n_th; j++) {
    thread_data[j].args = &op->args;
  }

  // load weights & bias scratch buffers (if necessary)
  size_t biases_src_offset = 0;
  size_t weights_src_offset = 0;
  size_t weights_fetch_size;

  if (op->weights_scratch_index >= 0) {
    op->args.K = static_cast<nn_tensor_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(op->args.K != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    op->args.BSO = static_cast<nn_bso_block_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(op->args.BSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.size(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    weights_fetch_size = weights_shape.Dims(1) * weights_shape.Dims(2) *
                         weights_shape.Dims(3) * changrp.size;
    dispatcher->FetchBuffer(
        (int8_t **)&op->args.K,
        &tflite::micro::GetTensorData<int8_t>(weights)[weights_src_offset],
        weights_fetch_size);
    weights_src_offset += weights_fetch_size;
    dispatcher->FetchBuffer(
        (int8_t **)&op->args.BSO,
        &tflite::micro::GetTensorData<int8_t>(bso)[biases_src_offset],
        bso_changrp_bytes);
    biases_src_offset += bso_changrp_bytes;

    // create tasks
    for (int i_rg = 0; i_rg < op->execution_plan.regions.size(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

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
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  const TfLiteEvalTensor *weights =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bso = tflite::micro::GetEvalInput(context, node, 2);

  const RuntimeShape weights_shape = tflite::micro::GetTensorShape(weights);

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
  for (int j{0}; j < n_th; j++) {
    thread_data[j].args = &op->args;
  }

  // load weights & bias scratch buffers (if necessary)
  size_t weights_src_offset = 0;
  size_t weights_fetch_size;
  size_t biases_src_offset = 0;

  if (op->weights_scratch_index >= 0) {
    op->args.K = static_cast<nn_tensor_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(op->args.K != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    op->args.BSO = static_cast<nn_bso_block_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(op->args.BSO != nullptr);
  }

  // create tasks
  for (int i_cg = 0; i_cg < op->execution_plan.changrps.size(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    weights_fetch_size = weights_shape.Dims(1) * weights_shape.Dims(2) *
                         weights_shape.Dims(3) * changrp.size;
    dispatcher->FetchBuffer(
        (int8_t **)&op->args.K,
        &tflite::micro::GetTensorData<int8_t>(weights)[weights_src_offset],
        weights_fetch_size);
    weights_src_offset += weights_fetch_size;
    dispatcher->FetchBuffer(
        (int8_t **)&op->args.BSO,
        &tflite::micro::GetTensorData<int8_t>(bso)[biases_src_offset],
        bso_changrp_bytes);
    biases_src_offset += bso_changrp_bytes;

    for (int i_rg = 0; i_rg < op->execution_plan.regions.size(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

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
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  const TfLiteEvalTensor *weights =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bso = tflite::micro::GetEvalInput(context, node, 2);

  const RuntimeShape weights_shape = tflite::micro::GetTensorShape(weights);

  Conv2DOpData *op = reinterpret_cast<Conv2DOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_1x1_thread_worker, stack, op->stack_size);

  // create thread data
  int n_th = op->execution_plan.GetNumThreads();
  Conv2DThreadData thread_data[n_th];
  for (int j{0}; j < n_th; j++) {
    thread_data[j].args = &op->args;
  }

  // load weights & bias scratch buffers (if necessary)
  size_t weights_src_offset = 0;
  size_t weights_fetch_size;
  size_t biases_src_offset = 0;

  if (op->weights_scratch_index >= 0) {
    op->args.K = static_cast<nn_tensor_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(op->args.K != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    op->args.BSO = static_cast<nn_bso_block_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(op->args.BSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.size(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    weights_fetch_size = weights_shape.Dims(1) * changrp.size;
    dispatcher->FetchBuffer(
        (int8_t **)&op->args.K,
        &tflite::micro::GetTensorData<int8_t>(weights)[weights_src_offset],
        weights_fetch_size);
    weights_src_offset += weights_fetch_size;
    dispatcher->FetchBuffer(
        (int8_t **)&op->args.BSO,
        &tflite::micro::GetTensorData<int8_t>(bso)[biases_src_offset],
        bso_changrp_bytes);
    biases_src_offset += bso_changrp_bytes;

    for (int i_rg = 0; i_rg < op->execution_plan.regions.size(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

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
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  auto *op = reinterpret_cast<Conv2DOpData *>(node->user_data);

  const TfLiteEvalTensor *weights =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bso = tflite::micro::GetEvalInput(context, node, 2);

  const RuntimeShape weights_shape = tflite::micro::GetTensorShape(weights);

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
  for (int j{0}; j < n_th; j++) {
    thread_data[j].args = &op->args;
  }

  // load weights & bias scratch buffers (if necessary)
  size_t biases_src_offset = 0;

  op->args.depthwise_flags = (op->weights_scratch_index >= 0)
                                 ? CONV2D_DEPTHWISE_FLAG_SLICED_K
                                 : (nn_conv2d_depthwise_flags_e)0;
  if (op->weights_scratch_index >= 0) {
    op->args.K = static_cast<nn_tensor_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(op->args.K != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    op->args.BSO = static_cast<nn_bso_block_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(op->args.BSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.size(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    if (op->weights_scratch_index >= 0) {
      // fetch the weights
      fetch_depthwise_subtensor(
          (int8_t *)op->args.K, tflite::micro::GetTensorData<int8_t>(weights),
          op->args.window.shape.height, op->args.window.shape.width,
          weights_shape.Dims(2), changrp.start, changrp.size);
    } else {
      // use entire tensor
      op->args.K = tflite::micro::GetTensorData<nn_tensor_t>(weights);
    }

    if (op->bias_scratch_index >= 0) {
      // fetch the biases
      dispatcher->FetchBuffer(
          (int8_t **)&op->args.BSO,
          &tflite::micro::GetTensorData<int8_t>(bso)[biases_src_offset],
          bso_changrp_bytes);
      biases_src_offset += bso_changrp_bytes;
    } else {
      // use entire tensor
      op->args.BSO = tflite::micro::GetTensorData<nn_bso_block_t>(bso);
    }

    for (int i_rg = 0; i_rg < op->execution_plan.regions.size(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];

      thread_data[i_rg].job = {{region.top, region.left, changrp.start},
                               {region.rows, region.cols, changrp.size}};
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
