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
    "exp_name": "maxent-run",
    "device": "cuda",
    "result_folder": "output",
    "mode": ["train", "test"],
    "statefile": None,
    ############################
    # data related params
    ############################
    "data.name": "adult",
    "data.val_size": 0,
    #################################
    # model params
    #################################
    "model.name": "maxent_arl",
    "model.z_size": 8,
    "model.arch_file": "src/arch/adult/adult_maxent_arl.py",
    "model.beta": 1.0,
    ################################
    # training params
    ###############################
    "train.batch_size": 128,
    "train.patience": 500,
    "train.max_epoch": 500,
    "train.lr": 1e-4,
    "train.weight_decay": 1e-4,
    "train.save_strategy": ["current", "last"],
    "train.evaluations": [],
    "train.stopping_criteria": None,
    "train.stopping_criteria_direction": "bigger",
    "train.log_every": 100,
    ##################################
    # testing params
    ##################################
    "test.batch_size": 100,
    "test.evaluations": [],
}
