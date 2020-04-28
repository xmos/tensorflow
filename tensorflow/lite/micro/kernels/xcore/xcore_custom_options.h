#ifndef XCORE_CUSTOM_OPTIONS_H_
#define XCORE_CUSTOM_OPTIONS_H_

#include "lib_ops/api/conv2d.h"
#include "lib_ops/api/par.h"
#include "lib_ops/api/pooling.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void parse_custom_options(const char *buffer, size_t length,
                          ::xcore::pooling::PoolingParams &pooling_params);

void parse_custom_options(const char *buffer, size_t length,
                          ::xcore::conv::Conv2DParams &conv2d_params,
                          ::xcore::ParRegionArray *par_regions = nullptr);

void parse_custom_options(const char *buffer, size_t length,
                          int32_t *stride_h = nullptr,
                          int32_t *stride_w = nullptr,
                          int32_t *pool_h = nullptr, int32_t *pool_w = nullptr,
                          int32_t *K_w = nullptr,
                          ::xcore::conv::Conv2DPadding *pad = nullptr,
                          ::xcore::ParRegionArray *par_regions = nullptr);

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_CUSTOM_OPTIONS_H_
