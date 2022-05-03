from box import Box

target_evaluations = Box({
    "num_runs": 5,
    "model_config": [
        Box({
            "name": "nn", "friendly_name": "nn_1_layer_normalized",
            "__nn__hidden_layer_sizes": (50,), "__nn__max_iter": 1000,
            "__nn__n_iter_no_change": 10, "preprocess": "normalize"
        }),
        Box({"name": "nn", "friendly_name": "nn_2_layer_normalized",
             "__nn__hidden_layer_sizes": (50, 50), "__nn__max_iter": 1000,
             "__nn__n_iter_no_change": 10, "preprocess": "normalize"
             }),
        Box({"name": "logreg", "friendly_name": "logistic_regression_normalized",
             "preprocess": "normalize", "__logreg__C": 1.0}),
        Box({"name": "rf", "friendly_name": "random_forest_normalized", "preprocess": "normalize"}),
        Box({"name": "svm", "friendly_name": "svm_normalized", "preprocess": "normalize"}),
        # ############################ Evaluation without pre-processing #######################
        # Box({
        #     "name": "nn", "friendly_name": "nn_1_layer",
        #     "__nn__hidden_layer_sizes": (50,), "__nn__max_iter": 1000,
        #     "__nn__n_iter_no_change": 10,
        # }),
        # # Box({"name": "nn", "friendly_name": "nn_2_layer",
        # #      "__nn__hidden_layer_sizes": (50, 50), "__nn__max_iter": 1000,
        # #      "__nn__n_iter_no_change": 10, "preprocess": "normalize"
        # #      }),
        # # Box({"name": "nn", "hidden_layer_sizes": tuple(), "max_iter": 1000,
        # #      "n_iter_no_change": 10, "friendly_name": "nn_0_layer",
        # #      "preprocess": "normalize"}),
        # Box({"name": "logreg", "friendly_name": "logistic_regression",
        #      "__logreg__C": 1.0}),
        # Box({"name": "rf", "friendly_name": "random_forest"}),
        # Box({"name": "svm", "friendly_name": "svm"}),
    ]
})
