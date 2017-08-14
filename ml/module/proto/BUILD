load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library_py", "tf_proto_library")

tf_proto_library(
    name = "estimator_config",
    srcs = [
        "estimator_config.proto",
    ],
    visibility = ["//ml:__subpackages__"],
)

tf_proto_library(
    name = "feature_column_schema",
    srcs = [
        "feature_column_schema.proto",
    ],
    protodeps = [
        "//tensorflow/core:all_protos",
    ],
    visibility = ["//ml:__subpackages__"],
)
