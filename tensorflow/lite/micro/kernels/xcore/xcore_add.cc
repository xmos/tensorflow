#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace add {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct BConv2DArguments {
  int8_t* Y;
  const int8_t* X0;
  const int8_t* X1;
  nn_add_params_t params;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //
struct AddThreadData {
  int32_t start;
  int32_t element_count;
  const BConv2DArguments* args;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void add_thread_worker(void* context) {
  auto* td = static_cast<AddThreadData*>(context);
  auto* args = td->args;
  add_elementwise(args->Y, args->X0, args->X1, &args->params, td->start,
                  td->element_count);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct AddOpData {
  BConv2DArguments args;
  PersistentArray<AddThreadData> threads;
  int stack_scratch_index = -1;
  size_t stack_size;
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = construct_persistent_object<AddOpData>(context);

  auto par_parser = CustomOptionParser(
      CustomOptionParser(buffer, length).parseNamedCustomOption("par").AsMap());
  auto job_sizes = par_parser.parseNamedCustomOption("eg").AsVector();
  auto n_threads = par_parser.parseNamedCustomOption("th").AsInt32();
  TFLITE_DCHECK_EQ(n_threads, job_sizes.size());

  // in this op we have one job per thread
  op_data->threads.allocate(context, n_threads);
  int start_idx = 0;
  for (int j{0}; j < n_threads; j++) {
    auto job_size = job_sizes[j].AsInt32();
    op_data->threads.append({start_idx, job_size, &op_data->args});
    start_idx = start_idx + job_size;
  }

  return op_data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto* op_data = reinterpret_cast<AddOpData*>(node->user_data);

  // TODO: memory map this instead
  const auto* bss = GetInput(context, node, 2);
  auto& params = op_data->args.params;
  params.input[0].shr = bss->data.i32[0];
  params.input[0].multiplier = bss->data.i32[1];
  params.input[1].shr = bss->data.i32[2];
  params.input[1].multiplier = bss->data.i32[3];
  params.output.bias = bss->data.i32[4];
  params.output.shr = bss->data.i32[5];

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op_data->stack_size, add_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->threads.size(),
      &op_data->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<AddOpData*>(node->user_data);

  op_data->args.X0 = tflite::micro::GetTensorData<int8_t>(
      tflite::micro::GetEvalInput(context, node, 0));
  op_data->args.X1 = tflite::micro::GetTensorData<int8_t>(
      tflite::micro::GetEvalInput(context, node, 1));
  op_data->args.Y = tflite::micro::GetTensorData<int8_t>(
      tflite::micro::GetEvalOutput(context, node, 0));

  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  auto* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);
  dispatcher->InitializeTasks(add_thread_worker, stack, op_data->stack_size);

  for (auto& thread : op_data->threads) {
    dispatcher->AddTask(reinterpret_cast<void*>(&thread));
  }
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

}  // namespace add

TfLiteRegistration* Register_Add_8() {
  static TfLiteRegistration r = {add::Init, nullptr, add::Prepare, add::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
