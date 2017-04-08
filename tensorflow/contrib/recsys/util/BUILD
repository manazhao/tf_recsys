py_library(
	name = "common",
	srcs_version = "PY2AND3",
	srcs = ["common.py"],
)

py_library(
	name = "feature_helper",
	srcs = ["feature_helper.py"],
	srcs_version = "PY2AND3",
	deps = [
		"//tensorflow:tensorflow_py"
		]
)

py_library(
	name = "load_data",
	srcs = ["load_data.py"],
	srcs_version = "PY2AND3",
	deps = [
		":feature_helper",
		"//tensorflow:tensorflow_py"
	]
)

py_test(
	name = "feature_helper_test",
	srcs = ["feature_helper_test.py"],
	srcs_version = "PY2AND3",
	size = "small",
	deps = [
		":common",
		":feature_helper"
	],
	args = [
		"--log_level=INFO"
	]
)

py_test(
	name = "load_data_test",
	srcs = ["load_data_test.py"],
	srcs_version = "PY2AND3",
	size = "small",
	deps = [
		":common",
		":load_data",
		"//tensorflow:tensorflow_py"
	],
	data = [
		"test_data/movie_sample.csv",
		"test_data/ratings_sample.csv"
	],
	args = [
		"--log_level=INFO"
	]
)