py_library(
	name = "common",
	srcs_version = "PY2AND3",
	srcs = ["common.py"],
)

py_library(
	name = "feature",
	srcs = ["feature.py"],
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
		":feature"
	]
)

py_test(
	name = "feature_test",
	srcs = ["feature_test.py"],
	srcs_version = "PY2AND3",
	size = "small",
	deps = [
		":common",
		":feature"
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
		":load_data"
	],
	data = [
		"test_data/movie_sample.csv",
		"test_data/ratings_sample.csv"
	],
	args = [
		"--log_level=INFO"
	]
)