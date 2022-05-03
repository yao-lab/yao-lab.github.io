dataset='adult'
epoch=400
beta=0.05
lambda_=0.1
cuda_id=4
config = {
    ##########################
    # wandb params
    ##########################
    "wandb.use": True,
    "wandb.run_id": None,
    "wandb.watch": False,
    ###########################
    # general params
    ###########################
    "dataset":dataset,
    "project": "invariance",
    "exp_name": "test",
    "device": "cuda:"+str(cuda_id),
    "result_folder": "result/"+dataset+"/fcrl",
    "mode": ["train", "test"],
    "statefile": None,
    ############################
    # data related params
    ############################
    "data.name": dataset+"_open",
    "data.val_size": 0,
    #################################
    # model params
    #################################
    "model.name": "fcrl",
    "model.z_size": 16,
    "model.latent_distribution": "gaussian",
    "model.arch_file": "src/arch/"+dataset+"/"+dataset+"_fcrl.py",
    "model.beta": beta,
    "model.lambda_": lambda_,
    ################################
    # training params
    ###############################
    "train.batch_size": 128,
    "train.patience": 20,
    "train.max_epoch": epoch,
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
