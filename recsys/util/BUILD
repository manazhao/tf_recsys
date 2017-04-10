package(
	default_visibility = ["//recsys:__subpackages__"]
)

py_library(
	name = "common",
	srcs_version = "PY2AND3",
	srcs = ["common.py"],
)

py_library(
	name = "dataset_helper",
	srcs = ["dataset_helper.py"],
	srcs_version = "PY2AND3",
	deps = [
		"//tensorflow:tensorflow_py",
	]
)

py_library(
	name = "feature_helper",
	srcs = ["feature_helper.py"],
	srcs_version = "PY2AND3",
	deps = [
		"//tensorflow:tensorflow_py",
		"//recsys/util/proto:config_proto_py",
		]
)

py_test(
	name = "dataset_helper_test",
	srcs = ["dataset_helper_test.py"],
	srcs_version = "PY2AND3",
	deps = [
		":common",
		":dataset_helper",
		":feature_helper",
	],
	args = [
		"--log_level=INFO"
	]
)

py_test(
	name = "feature_helper_test",
	srcs = ["feature_helper_test.py"],
	srcs_version = "PY2AND3",
	size = "small",
	deps = [
		":common",
		":feature_helper",
		"//recsys/util/proto:config_proto_py",
	],
	args = [
		"--log_level=INFO"
	]
)