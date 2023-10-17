from .average_meter import AverageMeter
from .accuracy import accuracy
from .knn_monitor import knn_monitor
from .logger import Logger
from .file_exist_fn import file_exist_check
import random
import torch
import numpy as np


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        print('Could not set cuda seed.')
        pass
