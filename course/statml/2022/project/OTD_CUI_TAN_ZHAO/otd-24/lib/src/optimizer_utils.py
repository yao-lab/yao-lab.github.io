from torch import optim, nn


def get_optimizer_scheduler(model, optimizer="adam", lr=1e-3, opt_params=None, scheduler=None,
                            scheduler_params=None):
    """
    scheduler_params:
        load_on_reduce : best/last/None (if best we load the best model in training so far)
            (for this to work, you should save the best model during training)
    """
    if scheduler_params is None:
        scheduler_params = {}
    if opt_params is None:
        opt_params = {}

    if isinstance(model, nn.Module):
        params = model.parameters()
    else:
        params = model
    if optimizer == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=opt_params["weight_decay"])
    elif optimizer == "sgd":
        optimizer = optim.SGD(params, lr=lr, weight_decay=opt_params["weight_decay"],
                              momentum=opt_params["momentum"], nesterov=True)
    else:
        raise Exception(f"{optimizer} not implemented")

    if scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=scheduler_params["gamma"],
                                              step_size=scheduler_params["step_size"])
    elif scheduler == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=scheduler_params["gamma"],
                                                   milestones=scheduler_params["milestones"])
    elif scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_params["T_max"])
    elif scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=scheduler_params["mode"],
                                                         patience=scheduler_params["patience"],
                                                         factor=scheduler_params["gamma"],
                                                         min_lr=1e-7, verbose=True,
                                                         threshold=1e-7)
    elif scheduler is None:
        scheduler = None
    else:
        raise Exception(f"{scheduler} is not implemented")

    if scheduler_params.get("load_on_reduce") is not None:
        setattr(scheduler, "load_on_reduce", scheduler_params.get("load_on_reduce"))
    return optimizer, scheduler
