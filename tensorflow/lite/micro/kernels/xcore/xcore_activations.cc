#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace activations {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* lut = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);

  lookup8(tflite::micro::GetTensorData<uint8_t>(output),
          tflite::micro::GetTensorData<uint8_t>(input),
          tflite::micro::GetTensorData<uint8_t>(lut), 0,
          input_shape.FlatSize());

  return kTfLiteOk;
}

}  // namespace activations

TfLiteRegistration* Register_Lookup_8() {
  static TfLiteRegistration r = {nullptr, nullptr, activations::Prepare,
                                 activations::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
