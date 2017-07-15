package(
	default_visibility = ["//tts:__subpackages__"]
)

py_library(
	name = "estimator_factory",
	srcs_version = "PY2AND3",
	srcs = ["estimator_factory.py"],
	deps = [
		"//tensorflow:tensorflow_py",
		"//tts/module/proto:estimator_config_py"
	]
)

py_test(
	name = "estimator_factory_test",
	srcs = ["estimator_factory_test.py"],
	srcs_version = "PY2AND3",
	data = [":testing_data/estimator_config.pbtext"],
	deps = [
		":estimator_factory",
		"//tts/common:flags",
		"//tts/module/proto:estimator_config_py"
	],
)
