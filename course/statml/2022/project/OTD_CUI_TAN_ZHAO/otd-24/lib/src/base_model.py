""" base model"""
import logging

import numpy as np
import torch.nn as nn

logger = logging.getLogger()


class Base(nn.Module):
    """ Base model with some util functions"""

    def stats(self, print_model=True):
        # print network model and information about parameters
        logger.info("Model info:::")
        if print_model:
            logger.info(self)
        count = 0
        for i in self.parameters():
            count += np.prod(i.shape)
        logger.info(f"Total parameters : {count}")

    def to(self, *args, **kwargs):
        if kwargs.get("device") is not None:
            self.device = kwargs.get("device")
        if len(args) > 0:
            self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError()
