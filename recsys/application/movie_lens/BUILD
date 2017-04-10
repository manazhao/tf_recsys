py_binary(
	name = "main",
	srcs = ["main.py"],
	srcs_version = "PY2AND3",
	deps = [
		":movie_lens_data",
		"//recsys/util:common"
	]
)

py_library(
	name = "movie_lens_data",
	srcs = ["movie_lens_data.py"],
	srcs_version = "PY2AND3",
	deps = [
		"//recsys/util:feature_helper",
		"//tensorflow:tensorflow_py",
	]
)


py_test(
	name = "movie_lens_data_test",
	srcs = ["movie_lens_data_test.py"],
	srcs_version = "PY2AND3",
	size = "small",
	deps = [
		"//recsys/util:common",
		":movie_lens_data",
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