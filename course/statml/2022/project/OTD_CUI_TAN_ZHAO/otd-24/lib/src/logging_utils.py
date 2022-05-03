import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()


def print_verbose(string, verbose):
    if verbose:
        print(string)


def loss_logger_helper(
        loss, aux_loss, writer: SummaryWriter, step: int, epoch: int, log_every: int,
        string: str = "train", force_print: bool = False, new_line: bool = False
):
    # write to tensorboard at every step but only print at log step or when force_print is passed
    writer.add_scalar(f"{string}/loss", loss, step)
    for k, v in aux_loss.items():
        writer.add_scalar(f"{string}/" + k, v, step)

    if step % log_every == 0 or force_print:
        logger.info(f"{string}/loss: ({step}/{epoch}) {loss}")

    if force_print:
        if new_line:
            for k, v in aux_loss.items():
                logger.info(f"{string}/{k}:{v} ")
        else:
            str_ = ""
            for k, v in aux_loss.items():
                str_ += f"{string}/{k}:{v} "
            logger.info(f"{str_}")
