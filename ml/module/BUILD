package(
	default_visibility = ["//ml:__subpackages__"]
)

py_library(
	name = "estimator_factory",
	srcs_version = "PY2AND3",
	srcs = ["estimator_factory.py"],
	deps = [
		"//tensorflow:tensorflow_py",
		"//ml/module/proto:estimator_config_py"
	]
)

py_library(
	name = "feature_column_factory",
	srcs_version = "PY2AND3",
	srcs = ["feature_column_factory.py"],
	deps = [
		"//tensorflow:tensorflow_py",
		"//ml/module/proto:feature_column_schema_py"
	]
)

py_test(
	name = "estimator_factory_test",
	srcs = ["estimator_factory_test.py"],
	srcs_version = "PY2AND3",
	data = [":testing_data/estimator_config.pbtxt"],
	deps = [
		":estimator_factory",
		"//ml/common:flags",
		"//ml/module/proto:estimator_config_py"
	],
)

py_test(
	name = "feature_column_factory_test",
	srcs = ["feature_column_factory_test.py"],
	srcs_version = "PY2AND3",
	data = [":testing_data/feature_column_schema.pbtxt"],
	deps = [
		":feature_column_factory",
		"//ml/common:flags",
		"//ml/module/proto:feature_column_schema_py"
	],
)
