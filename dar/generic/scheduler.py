"""
Module handling schedulers
"""

from torch.nn import Dropout, Dropout2d, Module
import numpy as np


class DropoutScheduler:
    """
    Class handling scheduler for dropout rate.
    """

    def __init__(self, models: dict[str, Module], function: callable, T: int=1e5):
        """
        :param models: the models on which to apply dropout scheduling represented as a dict with model names as keys, and pytorch modules as values.
        :param function: the scheduler function to apply.
        :param T: number of gradient updates to converge.
        """

        self.teta_list = []
        self.modules_with_dropout_param = []
        self.init_teta_list(models)
        self.function = function
        self.T = T
        self.step_num = 0

    def step(self) -> None:
        """
        Increment by 1 the current step
        """
        self.step(1)

    def step(self, num: int) -> None:
        """
        Increment by num the current step
        """
        self.step_num += num

    def init_teta_list(self, models:dict[str, Module]) -> None:
        """
        Store list of layers impacted by dropout scheduling, as well as dropout rate (for given models)
        """
        for model_name in models.keys():
            self.init_teta_list_module(models[model_name])

    def init_teta_list_module(self, module: Module) -> None:
        """
        Store list of layers impacted by dropout scheduling, as well as dropout rate (for given module)
        """
        for child in module.children():
            if isinstance(child, Dropout) or isinstance(child, Dropout2d):
                self.teta_list.append([child, child.p])
            elif hasattr(child, "dropout") and isinstance(getattr(child, "dropout"), float):
                self.modules_with_dropout_param.append([child, child.dropout])
            else:
                self.init_teta_list_module(child)

    def update_dropout_rate(self) -> None:
        """
        Update dropout rate for all impacted layers with respect to current step.
        """
        for (module, p) in self.teta_list:
            module.p = self.function(p, self.step_num, self.T)
        for (module, p) in self.modules_with_dropout_param:
            module.dropout = self.function(p, self.step_num, self.T)


def exponential_dropout_scheduler(dropout_rate: float, step: int, max_step: int) -> float:
    """
    Compute dropout rate value at given step for exponential evolution from 0 to dropout_rate over max_step iterations
    """
    return dropout_rate * (1 - np.exp(-10 * step / max_step))


def exponential_scheduler(init_value: float, end_value: float, step: int, max_step: int) -> float:
    """
    Compute value at given step for exponential evolution from begin_value to end_value over max_step iterations
    """
    step = min(step, max_step-1)
    return init_value - (init_value - end_value) * (1 - np.exp(-10*step/max_step))


def linear_scheduler(init_value: float, end_value: float, step: int, max_step: int) -> None:
    """
    Compute value at given step for linear evolution from begin_value to end_value over max_step iterations
    """
    return init_value + step * (end_value - init_value) / max_step