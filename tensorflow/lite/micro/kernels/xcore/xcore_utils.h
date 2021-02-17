#ifndef XCORE_UTILS_H_
#define XCORE_UTILS_H_

#include <cassert>
#include <cstdint>
#include <utility>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

/* Get size (in bytes) given a TfLiteType enum
 *  This is useful because a tensor's type field is a TfLiteType
 *
 *  Returns kTfLiteError if the type is not supported
 *
 *  NOTE: This is cribbed from tensorflow/lite/util.h because TFLu does not
 * fully support the methods defined in tensorflow/lite/util.h
 */
TfLiteStatus GetSizeOfType(TfLiteContext *context, const TfLiteType type,
                           size_t *bytes);

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

template <typename T>
static inline T *construct_persistent_object(TfLiteContext *context) {
  return new (context->AllocatePersistentBuffer(context, sizeof(T))) T;
}

static inline bool is_ram_address(uintptr_t a) {
#ifdef XCORE
  return ((a >= 0x80000) && (a <= 0x100000));
#else
  return true;
#endif
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

template <typename T>
class PersistentArray {
 private:
  size_t max_size_ = 0;
  size_t size_ = 0;
  T *data_ = nullptr;

 public:
  // call this only in the Init phase of operators
  PersistentArray<T> &allocate(TfLiteContext *context,
                               size_t max_size) noexcept {
    assert(data_ == nullptr);
    assert(max_size > 0);

    max_size_ = max_size;
    data_ = reinterpret_cast<T *>(
        context->AllocatePersistentBuffer(context, sizeof(T) * max_size));

    return *this;
  };
  PersistentArray<T> &initialize() noexcept {
    assert(size_ == 0);
    while (size_ < max_size_) {
      this->append(T());
    }

    return *this;
  };
  // TODO: begin and end would be better if returned an iterator object
  inline T *begin() noexcept {
    assert(size_ > 0);
    return &data_[0];
  }
  inline T *end() noexcept {
    assert(size_ > 0);
    return &data_[size_];
  }
  inline T &operator[](int i) noexcept {
    assert(i < size_);
    return data_[i];
  }
  inline void append(const T &element) noexcept {
    assert(size_ < max_size_);
    data_[size_++] = element;
  }
  inline void append(T &&element) noexcept {
    assert(size_ < max_size_);
    data_[size_++] = std::move(element);
  }
  inline size_t size() noexcept { return size_; }
  inline size_t max_size() noexcept { return max_size_; }
};

#ifndef UNSUPPORTED_KERNEL_TYPE
#define UNSUPPORTED_KERNEL_TYPE(T) TF_LITE_FATAL("Unsupported " #T " value")
#endif /*UNSUPPORTED_KERNEL_TYPE*/

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_UTILS_H_