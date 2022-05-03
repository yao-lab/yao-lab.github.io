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
    "data.val_size": 0,
    #################################
    # model params
    #################################
    "model.name": "cvae_cc_supervised",
    "model.z_size": 8,
    "model.latent_distribution": "gaussian",
    "model.output_distribution": "bernoulli",
    "model.arch_file": "src/arch/adult/adult_cvae_cc_supervised.py",
    "model.beta": 1.0,
    "model.lambda_": 1.0,
    ################################
    # training params
    ###############################
    "train.batch_size": 128,
    "train.patience": 500,
    "train.max_epoch": 500,
    "train.optimizer": "adam",
    "train.lr": 1e-3,
    "train.weight_decay": 1e-4,
    "train.save_strategy": ["best", "last", "current"],
    "train.log_every": 100,
    "train.stopping_criteria": "loss",
    "train.stopping_criteria_direction": "lower",
    "train.evaluations": [],
    ##################################
    # testing params
    ##################################
    "test.batch_size": 100,
    "test.eval_model": "last",
    "test.evaluations": [],
}
