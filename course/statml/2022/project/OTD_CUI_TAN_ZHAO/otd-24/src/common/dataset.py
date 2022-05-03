import logging

import torch
from box import Box
from torch.utils.data import TensorDataset

from src.common.data.adult import load_adult
from src.common.data.health import load_health
from src.common.data.adult import load_adult_open
from src.common.data.health import load_health_open
logger = logging.getLogger()


def get_dataset(data_config, device="cpu"):
    """ Take config and return dataloader"""
    dataname = data_config.name
    val_size = data_config.val_size

    if dataname == "adult_open":
        data = load_adult_open(val_size=val_size)
        c_size = [1,3,1,8,1]
        c_mix_num=31
        c_type = "open"
    elif dataname == "health_open":
        data = load_health_open(val_size=val_size)
        c_size = [9,1]
        c_mix_num=3
        c_type = "open"
    elif dataname == "adult":
        data = load_adult(val_size=val_size)
        c_size = 2
        c_type = "binary"
    else:
        logger.error(f"Invalid data name {dataname} specified")
        raise Exception(f"Invalid data name {dataname} specified")

    train, valid, test = data["train"], data["valid"], data["test"]
    if valid is None:
        valid = data["test"]

    return (
        Box({"train": TensorDataset(
            torch.tensor(train[0]).float().to(device),
            torch.tensor(train[1]).long().to(device),
            torch.tensor(train[2]).long().to(device),
            torch.tensor(train[3]).long().to(device),

        ), "test": TensorDataset(
            torch.tensor(test[0]).float().to(device),
            torch.tensor(test[1]).long().to(device),
            torch.tensor(test[2]).long().to(device),
            torch.tensor(test[3]).long().to(device),

        ), "valid": TensorDataset(
            torch.tensor(valid[0]).float().to(device),
            torch.tensor(valid[1]).long().to(device),
            torch.tensor(valid[2]).long().to(device),
            torch.tensor(valid[3]).long().to(device),

        )}), {
            "input_shape": train[0].shape[1:],
            "c_size": c_size,
            "c_mix_num": c_mix_num,
            "c_type": c_type,
            "y_size": 2,
            "y_type": "binary",
        })
