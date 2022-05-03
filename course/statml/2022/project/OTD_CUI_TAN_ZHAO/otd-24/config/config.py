# Example config

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
    #    .name:  possible options : adult, health
    ############################
    "data.name": "adult",
    "data.val_size": 0,
    #################################
    # model params
    #    .name : name of the model
    #    .z_size : latent space size for representations
    #    .latent_distribution : "gaussian"
    #    .output_distribution : "bernoulli"/"gaussian"
    #    .arch_file : To see what class/def to implement see arch/<data>/<model name>.py file
    #    .use_posterior_sample: whether to use posterior samples or mu for supervised classification
    #    in cvib
    #
    # NOTE: Ideally here you could move model params to model.<model_name>.<param>
    # but that might just be too much. So we define params at the same level,
    # these gives freedom to use params with same name in multiple model
    #################################
    "model.name": "cvae",
    "model.z_size": 32,
    "model.latent_distribution": "gaussian",
    "model.output_distribution": "bernoulli",
    "model.arch_file": None,
    "model.beta": 1.0,
    "model.lambda_": 1.0,
    "model.rho": 1.0,
    "model.delta": 1.0,
    "model.use_posterior_sample": True,
    ################################
    # training params
    #    .lr : learning rate
    #    .save_strategy: what strategy to use for saving
    #           "best": keep the best performing model on validation set
    #           "last": keep the last model during training
    #           "epoch": save model every epoch
    #    .train_evaluations: evaluations that should be done during training
    ###############################
    "train.batch_size": 100,
    "train.patience": 10,
    "train.max_epoch": 200,
    "train.lr": 3e-4,
    "train.weight_decay": 0.0,
    "train.save_strategy": ["best", "last"],
    "train.evaluations": [],
    "train.model_update_freq": 11,
    "train.stopping_criteria": "acc",
    "train.stopping_criteria_direction": "bigger",
    "train.log_every": 100,
    ##################################
    # testing params
    #    .batch_size :  batch_size to use for testing
    #    .evaluations : final evaluations to be done on train and test set
    #                   This could have lot more evaluations and visualizations
    #                     as it is done only once
    ##################################
    "test.batch_size": 100,
    "test.evaluations": [],
}
