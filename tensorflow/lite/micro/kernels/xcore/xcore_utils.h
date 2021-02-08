#ifndef XCORE_UTILS_H_
#define XCORE_UTILS_H_

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

/* Unpack an integer data type from a byte array
 *  T  data type to unpack
 *
 * Example usage:
 *      int32_t t0 = unpack<int32_t>(&my_buffer[23]);
 *      int32_t t1 = unpack<int32_t>(&my_buffer[27]);
 */
template <class T>
T unpack(const uint8_t *buffer) {
  T retval = 0;
  for (int i = 0; i < sizeof(T); ++i) retval |= buffer[i] << (8 * i);
  return retval;
}

static inline TfLiteStatus request_scratch_if_needed(TfLiteContext *context,
                                                     const TfLiteTensor *tensor,
                                                     int &scratch_idx) {
  if (!is_ram_address((uintptr_t)tensor->data.data)) {
    return context->RequestScratchBufferInArena(context, tensor->bytes,
                                                &scratch_idx);
  }
  return kTfLiteOk;
}

#ifndef UNSUPPORTED_KERNEL_TYPE
#define UNSUPPORTED_KERNEL_TYPE(T) TF_LITE_FATAL("Unsupported " #T " value")
#endif /*UNSUPPORTED_KERNEL_TYPE*/

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_UTILS_H_