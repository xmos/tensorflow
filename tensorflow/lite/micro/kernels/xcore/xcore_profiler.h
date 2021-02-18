// Copyright (c) 2019, XMOS Ltd, All rights reserved

#ifndef XCORE_PROFILER_H_
#define XCORE_PROFILER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_profiler.h"

#if !defined(XCORE_PROFILER_MAX_LEVELS)
#define XCORE_PROFILER_MAX_LEVELS (64)
#endif

namespace tflite {
namespace micro {
namespace xcore {

class XCoreProfiler : public tflite::MicroProfiler {
 public:
  explicit XCoreProfiler();
  ~XCoreProfiler() override = default;

  uint32_t BeginEvent(const char* tag) override;

  // Event_handle is ignored since TFLu does not support concurrent events.
  void EndEvent(uint32_t event_handle) override;

  uint32_t const* GetEventDurations();
  uint32_t GetNumEvents();

 private:
  const char* event_tag_;
  uint32_t event_start_time_;
  uint32_t event_count_;
  uint32_t event_durations_[XCORE_PROFILER_MAX_LEVELS];
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_PROFILER_H_
