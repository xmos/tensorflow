# Description:
#   TensorFlow Lite for Microcontrollers "hello xmos" example.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "mnist_model_data",
    srcs = [
        "mnist_model_data.cc",
    ],
    hdrs = [
        "mnist_model_data.h",
    ],
)

tflite_micro_cc_test(
    name = "mnist_digits_test",
    srcs = [
        "hello_world_test.cc",
    ],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/mnist_digits:mnist_model_data",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/experimental/micro/kernels:micro_ops",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "output_handler",
    srcs = [
        "output_handler.cc",
    ],
    hdrs = [
        "output_handler.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)

cc_library(
    name = "constants",
    srcs = [
        "constants.cc",
    ],
    hdrs = [
        "constants.h",
    ],
)

cc_binary(
    name = "mnist_digits",
    srcs = [
        "main.cc",
    ],
    copts = [
        "-Werror",
        "-Wdouble-promotion",
        "-Wsign-compare",
    ],
    deps = [
        ":constants",
        ":output_handler",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/experimental/micro:micro_framework",
        "//tensorflow/lite/experimental/micro/examples/mnist_digits:mnist_model_data",
        "//tensorflow/lite/experimental/micro/kernels:all_ops_resolver",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
