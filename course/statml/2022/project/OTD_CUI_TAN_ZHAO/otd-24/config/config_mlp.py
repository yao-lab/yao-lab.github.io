config = {
    ##########################
    # wandb params
    ##########################
    "wandb.use": False,
    "wandb.run_id": None,
    "wandb.watch": False,
    ###########################
    # general params
    ###########################
    "project": "invariance",
    "exp_name": "Test run",
    "device": "cuda",
    "result_folder": "output",
    "mode": ["train", "test"],
    "statefile": None,
    ############################
    # data related params
    ############################
    "data.name": "adult",
    "data.val_size" : 0.2,
    # for mlp we will use a 80-20 split like we do while training classifier on representations
    #################################
    # model params
    #################################
    "model.name": "mlp",
    "model.arch_file": "src/arch/mlp.py",
    ################################
    # training params
    ###############################
    "train.batch_size": 128,
    "train.patience": 500,
    "train.max_epoch": 500,
    "train.lr": 1e-3,
    "train.weight_decay": 1e-4,
    "train.save_strategy": ["best", "last"],
    "train.evaluations": [],
    "train.stopping_criteria": "accuracy",
    "train.stopping_criteria_direction": "bigger",
    "train.log_every": 100,
    ##################################
    # testing params
    ##################################
    "test.batch_size": 100,
    "test.eval_model": "best",
    "test.evaluations": [],
}
