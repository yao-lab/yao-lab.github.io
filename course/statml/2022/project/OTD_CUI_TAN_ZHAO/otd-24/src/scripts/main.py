"""entry point for training a classifier"""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append("/home/data/cuipaofu/otd-24/")
import argparse
import importlib
import json
import logging
import pprint


import dill
import torch
import wandb
from box import Box
from torch.utils.data import DataLoader
from datetime import datetime

from lib import logging_utils
from lib import os_utils
from lib.src import optimizer_utils
from src.common.dataset import get_dataset


def parser_setup():
    # define argparsers
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--config", "-c", default="../../config/config_fcrl.py")
    parser.add_argument("--dataset", default="health")

    str2bool = os_utils.str2bool
    parser.add_argument("--wandb.use", required=False, type=str2bool, default=False)
    parser.add_argument("--wandb.run_id", required=False, type=str)
    parser.add_argument("--wandb.watch", required=False, type=str2bool, default=False)
    parser.add_argument("--project", required=False, type=str, default="generative-invariance")
    parser.add_argument("--exp_name", required=False,default='test')

    parser.add_argument("--device", required=False,
                        default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result_folder", "-r", default="result/health/fcrl",required=False)
    parser.add_argument("--mode", required=False, nargs="+", choices=["test", "train"],
                        default=["test", "train"])
    parser.add_argument("--statefile", "-s", required=False, default=None)

    parser.add_argument("--data.name", "-d", default="health_open",required=False, choices=["adult", "health","adult_open", "health_open"])
    parser.add_argument("--data.val_size", default=0, type=float, required=False)

    parser.add_argument("--model.name", required=False,
                        choices=["cvae_cc_supervised", "cvae_supervised", "cvib_supervised",
                                 "adv_forgetting", "mlp", "fcrl", "fcrl_no_conditioning",
                                 "maxent_arl"]
                        )
    parser.add_argument("--model.z_size", required=False, type=int),
    parser.add_argument("--model.latent_distribution", required=False, choices=["gaussian", "echo"])
    parser.add_argument("--model.output_distribution", required=False,
                        choices=["bernoulli", "gaussian"])
    parser.add_argument("--model.arch_file", default="src/arch/health/health_fcrl.py",required=False, type=str)
    parser.add_argument("--model.beta", default=0.001,required=False, type=float)
    parser.add_argument("--model.lambda_", default=2,required=False, type=float)
    parser.add_argument("--model.rho", required=False, type=float)
    parser.add_argument("--model.delta", required=False, type=float)
    parser.add_argument("--model.use_posterior_sample", required=False, type=str2bool)

    parser.add_argument("--train.batch_size", required=False, type=int, default=128)
    parser.add_argument("--train.patience", required=False, type=int, default=20)
    parser.add_argument("--train.max_epoch", required=False, type=int, default=50)
    parser.add_argument("--train.optimizer", required=False, type=str, default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--train.lr", required=False, type=float, default=1e-3)
    parser.add_argument("--train.weight_decay", required=False, type=float, default=5e-4)
    parser.add_argument("--train.model_update_freq", required=False, type=int)

    parser.add_argument("--train.save_strategy", required=False, nargs="+",
                        choices=["best", "last", "init", "epoch", "current"],
                        default=["best", "current"])
    parser.add_argument("--train.log_every", required=False, type=int, default=1000)
    parser.add_argument("--train.stopping_criteria", required=False, type=str, default="accuracy")
    parser.add_argument("--train.stopping_criteria_direction", required=False,
                        choices=["bigger", "lower"], default="bigger")
    parser.add_argument("--train.evaluations", required=False, nargs="*", choices=[])

    #
    parser.add_argument("--test.batch_size", required=False, type=int, default=128)
    parser.add_argument("--test.evaluations", required=False, nargs="*", choices=[])
    parser.add_argument("--test.eval_model", required=False, type=str,
                        choices=["best", "last", "current"])

    return parser
#-c ../../config/config_fcrl.py --exp_name test   --data.name health_open --model.arch_file  src/arch/health/health_fcrl.py      --result_folder result/health/fcrl --train.max_epoch 100 --device cuda:1  --model.lambda_ 2 --model.beta 0.001

if __name__ == "__main__":
    # set seeds etc here
    lambda_id=0 #0,1,2,3
    beta_id=1 #0,1
    lambda_set=[[1e-3,  3e-3, 5e-3, 7e-3,  9e-3],[1e-2,  3e-2, 5e-2, 7e-2,  9e-2, 1e-1],[ 2e-1,  4e-1, 6e-1,  8e-1, 1e0],[ 11e-1,  13e-1, 15e-1,  17e-1,  19e-1, 2e0]]
    beta_set=[[1e-4,1e-3,1e-2], [1e-1,1e0, 2e0]]
    for lambda_ in lambda_set[lambda_id]:
        for  beta in beta_set[beta_id]:

            torch.backends.cudnn.benchmark = True

            # define logger etc
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
            logger = logging.getLogger()

            parser = parser_setup()
            config = os_utils.parse_args(parser)
            if config.debug:
                logger.setLevel(logging.DEBUG)

            logger.info("Config:")
            logger.info(pprint.pformat(config.to_dict(), indent=4))

            # see https://github.com/wandb/client/issues/714
            os_utils.safe_makedirs(config.result_folder)
            statefile, run_id, result_folder = os_utils.get_state_params(
                config.wandb.use, config.wandb.run_id, config.result_folder, config.statefile
            )
            config.statefile = statefile
            config.wandb.run_id = run_id
            config.result_folder = result_folder
            config.model.lambda_=lambda_
            config.model.beta=beta

            if statefile is not None:
                data = torch.load(open(statefile, "rb"), pickle_module=dill)
                epoch = data["epoch"]
                if epoch >= config.train.max_epoch:
                    logger.error("Aleady trained upto max epoch; exiting")
                    sys.exit()
            now= datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            config.name='otd'+"_"+config.data.name+"_"+now
            if config.wandb.use:
                wandb.init(
                    project="OTD_"+config.dataset,config=config.to_dict(),name=config.name,notes='x->y')

                logger.info(f"Starting wandb with id {wandb.run.id}")

            # NOTE: WANDB creates git patch so we probably can get rid of this in future
            # os_utils.copy_code("src", config.result_folder, replace=True)
            # json.dump(
            #     config.to_dict(),
            #     open(f"{wandb.run.dir if config.wandb.use else config.result_folder}/config.json", "w"))

            logger.info("Getting data and dataloaders")
            #data: the train val test in Boxes
            # meta: descriptions on the meta-information about the dataset
            data, meta = get_dataset(config.data, config.device)

            num_workers = 0  # max(min(os.cpu_count(), 8), 1)
            logger.info(f"Using {num_workers} workers")
            train_loader = DataLoader(data["train"], shuffle=True, batch_size=config.train.batch_size,
                                      num_workers=num_workers)
            valid_loader = DataLoader(data["valid"], shuffle=False, batch_size=config.test.batch_size,
                                      num_workers=num_workers)
            test_loader = DataLoader(data["test"], shuffle=False, batch_size=config.test.batch_size,
                                     num_workers=num_workers)

            logger.info("Getting model")
            # load arch module
            arch_module = importlib.import_module(config.model.arch_file.replace("/", ".")[:-3])
            model_arch = arch_module.get_arch(
                input_size=meta["input_shape"],
                z_size=config.model.z_size,
                y_size=meta["y_size"],
                y_type=meta["y_type"],
                c_size=meta["c_size"],
                c_type=meta["c_type"]
            )

            # declaring models
            if config.model.name in ["cvae_supervised", "cvae_cc_supervised"]:
                if config.model.name == "cvae_supervised":
                    from src.models.cvae_supervised_model import CVAESupervised as Model
                elif config.model.name == "cvae_cc_supervised":
                    from src.models.cvae_cc_supervised_model import CVAEccSupervised as Model

                model = Model(**model_arch, c_type=meta["c_type"], y_type=meta["y_type"],
                              c_size=meta["c_size"], y_size=meta["y_size"], z_size=config.model.z_size,
                              beta=config.model.beta, lambda_=config.model.lambda_,
                              latent_distribution=config.model.latent_distribution,
                              output_distribution=config.model.output_distribution,
                              )

            elif config.model.name in ["fcrl_no_conditioning", "fcrl"]:
                if config.model.name == "fcrl_no_conditioning":
                    from src.models.fcrl_no_conditioning_model import FCRL_NO_Conditioning as Model
                elif config.model.name == "fcrl":
                    from src.models.fcrl_model import FCRL as Model

                model = Model(**model_arch, c_type=meta["c_type"], c_size=meta["c_size"],c_mix_num=meta["c_mix_num"],
                              y_type=meta["y_type"], y_size=meta["y_size"],
                              latent_distribution=config.model.latent_distribution,
                              beta=config.model.beta, lambda_=config.model.lambda_,
                              z_size=config.model.z_size)

            elif config.model.name == "cvib_supervised":
                from src.models.cvib_supervised_model import CVIBSupervised as Model

                model = Model(**model_arch, c_type=meta["c_type"], y_type=meta["y_type"],
                              c_size=meta["c_size"], y_size=meta["y_size"], z_size=config.model.z_size,
                              beta=config.model.beta, lambda_=config.model.lambda_,
                              use_posterior_sample=config.model.use_posterior_sample,
                              latent_distribution=config.model.latent_distribution,
                              output_distribution=config.model.output_distribution)

            elif config.model.name == "mlp":
                from src.models.mlp_model import MLP

                model = MLP(**model_arch, y_type=meta["y_type"])

            elif config.model.name == "adv_forgetting":
                from src.models.adv_forgetting_model import AdvForgetting

                model = AdvForgetting(**model_arch, c_type=meta["c_type"], y_type=meta["y_type"],
                                      c_size=meta["c_size"], y_size=meta["y_size"], rho=config.model.rho,
                                      delta=config.model.delta, lambda_=config.model.lambda_,
                                      z_size=config.model.z_size)

            elif config.model.name == "maxent_arl":
                from src.models.maxent_arl_model import MaxentARL

                model = MaxentARL(**model_arch, z_size=config.model.z_size, beta=config.model.beta,
                                  c_type=meta["c_type"], c_size=meta["c_size"], y_size=meta["y_size"],
                                  y_type=meta["y_type"])

            else:
                raise Exception(f"Model {config.model.name} not found")

            model.to(config.device)
            model.stats()

            if config.wandb.use and config.wandb.watch:
                wandb.watch(model, log="all")

            # declaring trainer
            if config.model.name == "adv_forgetting":
                from src.trainers.adv_forgetting import AdvForgettingTrainer

                # no scheduler for adversarial model
                disc_optimizer, _ = optimizer_utils.get_optimizer_scheduler(
                    model._discriminator.parameters(),
                    lr=config.train.lr,
                    optimizer=config.train.optimizer,
                    opt_params={
                        "weight_decay": config.train.get("weight_decay", 1e-4),
                        "momentum": config.train.get("optimizer_momentum", 0.9)
                    }
                )

                model_optimizer, _ = optimizer_utils.get_optimizer_scheduler([
                    {"params": model._encoder.parameters()},
                    {"params": model._decoder.parameters()},
                    {"params": model._predictor.parameters()},
                    {"params": model._mask.parameters()}
                ],
                    lr=config.train.lr,
                    optimizer=config.train.optimizer,
                    opt_params={
                        "weight_decay": config.train.get("weight_decay", 1e-4),
                        "momentum": config.train.get("optimizer_momentum", 0.9)
                    }
                )
                optimizer = Box({"disc": disc_optimizer, "model": model_optimizer})
                trainer = AdvForgettingTrainer(model, optimizer, scheduler=None,
                                               statefile=config.statefile,
                                               result_dir=config.result_folder,
                                               log_every=config.train.log_every,
                                               save_strategy=config.train.save_strategy,
                                               patience=config.train.patience,
                                               max_epoch=config.train.max_epoch, stopping_criteria=None,
                                               evaluations=Box({"train": config.train.evaluations,
                                                                "test": config.test.evaluations}),
                                               model_update_freq=config.train.model_update_freq)
            elif config.model.name == "maxent_arl":
                from src.trainers.maxent_arl import MaxentARLTrainer

                # no scheduler for adversarial model
                disc_optimizer, _ = optimizer_utils.get_optimizer_scheduler(
                    model._discriminator.parameters(),
                    lr=config.train.lr,
                    optimizer=config.train.optimizer,
                    opt_params={
                        "weight_decay": config.train.get("weight_decay", 1e-4),
                        "momentum": config.train.get("optimizer_momentum", 0.9)
                    }
                )

                model_optimizer, _ = optimizer_utils.get_optimizer_scheduler([
                    {"params": model._encoder.parameters()},
                    {"params": model._predictor.parameters()},
                ],
                    lr=config.train.lr,
                    optimizer=config.train.optimizer,
                    opt_params={
                        "weight_decay": config.train.get("weight_decay", 1e-4),
                        "momentum": config.train.get("optimizer_momentum", 0.9)
                    }
                )
                optimizer = Box({"disc": disc_optimizer, "model": model_optimizer})
                trainer = MaxentARLTrainer(model, optimizer, scheduler=None,
                                           statefile=config.statefile,
                                           result_dir=config.result_folder,
                                           log_every=config.train.log_every,
                                           save_strategy=config.train.save_strategy,
                                           patience=config.train.patience,
                                           max_epoch=config.train.max_epoch, stopping_criteria=None,
                                           evaluations=Box({"train": config.train.evaluations,
                                                            "test": config.test.evaluations}))

            elif config.model.name in ["cvib_supervised", "cvae_supervised", "fcrl_no_conditioning",
                                       "fcrl", "cvae_cc_supervised"]:
                from src.trainers.fcrl import FCRLTrainer

                # no scheduler for adversarial model
                optimizer, scheduler = optimizer_utils.get_optimizer_scheduler(
                    model,
                    lr=config.train.lr,
                    optimizer=config.train.optimizer,
                    opt_params={
                        "weight_decay": config.train.get("weight_decay", 1e-4),
                        "momentum": config.train.get("optimizer_momentum", 0.9)
                    },
                    scheduler=config.train.get("scheduler", None),
                    scheduler_params={
                        "gamma": config.train.get("scheduler_gamma", 0.1),
                        "milestones": config.train.get("scheduler_milestones", [100, 200, 300]),
                        "patience": config.train.get("scheduler_patience", 100),
                        "step_size": config.train.get("scheduler_step_size", 100),
                        "load_on_reduce": config.train.get("scheduler_load_on_reduce"),
                        "mode": "max" if config.train.get(
                            "stopping_criteria_direction") == "bigger" else "min"
                    },
                )
                trainer = FCRLTrainer(model, optimizer, scheduler=scheduler, statefile=config.statefile,
                                      result_dir=config.result_folder, log_every=config.train.log_every,
                                      save_strategy=config.train.save_strategy,
                                      patience=config.train.patience,
                                      max_epoch=config.train.max_epoch,
                                      stopping_criteria=config.train.stopping_criteria,
                                      stopping_criteria_direction=config.train.stopping_criteria_direction,
                                      evaluations=Box({"train": config.train.evaluations,
                                                       "test": config.test.evaluations}))

            elif config.model.name == "mlp":
                from src.trainers.mlp import MLPTrainer

                # no scheduler for adversarial model
                optimizer, scheduler = optimizer_utils.get_optimizer_scheduler(
                    model,
                    lr=config.train.lr,
                    optimizer=config.train.optimizer,
                    opt_params={
                        "weight_decay": config.train.get("weight_decay", 1e-4),
                        "momentum": config.train.get("optimizer_momentum", 0.9)
                    },
                    scheduler=config.train.get("scheduler", None),
                    scheduler_params={
                        "gamma": config.train.get("scheduler_gamma", 0.1),
                        "milestones": config.train.get("scheduler_milestones", [100, 200, 300]),
                        "patience": config.train.get("scheduler_patience", 100),
                        "step_size": config.train.get("scheduler_step_size", 100),
                        "load_on_reduce": config.train.get("scheduler_load_on_reduce"),
                        "mode": "max" if config.train.get(
                            "stopping_criteria_direction") == "bigger" else "min"
                    },
                )

                trainer = MLPTrainer(model, optimizer, scheduler=scheduler, statefile=config.statefile,
                                     result_dir=config.result_folder, log_every=config.train.log_every,
                                     save_strategy=config.train.save_strategy,
                                     patience=config.train.patience,
                                     max_epoch=config.train.max_epoch,
                                     stopping_criteria=config.train.stopping_criteria,
                                     stopping_criteria_direction=config.train.stopping_criteria_direction,
                                     evaluations=Box({"train": config.train.evaluations,
                                                      "test": config.test.evaluations}))

            if "train" in config.mode:
                logger.info("starting training")
                trainer.train(train_loader, valid_loader)
                logger.info("Training done;")

                # copy current step and write test results to
                step_to_write = trainer.step
                step_to_write += 1

                if "test" in config.mode and config.test.eval_model == "best":
                    if os.path.exists(f"{trainer.result_dir}/best_model.pt"):
                        logger.info("Loading best model")
                        trainer.load(f"{trainer.result_dir}/best_model.pt")
                    else:
                        logger.info("eval_model is best, but best model not found ::: evaling last model")
                else:
                    logger.info("eval model is not best, so skipping loading at end of training")

            if "test" in config.mode:
                logger.info("evaluating model on test set")
                logger.info(f"Model was trained upto {trainer.epoch}")
                # copy current step and write test results to
                step_to_write = trainer.step
                step_to_write += 1
                loss, aux_loss = trainer.test(train_loader, test_loader)
                logging_utils.loss_logger_helper(loss, aux_loss, writer=trainer.summary_writer,
                                                 force_print=True, step=step_to_write,
                                                 epoch=trainer.epoch,
                                                 log_every=trainer.log_every, string="test",
                                                 new_line=True)
                if config.wandb.use:
                    wandb.log(aux_loss)
                    wandb.finish()