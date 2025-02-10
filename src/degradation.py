import typing
from dataclasses import dataclass

import numpy as np

from src.environment_typing import envt


@dataclass
class DegradationModel:
    alpha: envt.float
    beta: envt.float

    def percent_degradation(self, depth_of_discharge):
        stress_function_degradation = self.__alpha * np.exp(
            -self.__beta * depth_of_discharge
        )
        return stress_function_degradation

    @classmethod
    def default(cls) -> typing.Self:
        alpha = 0.4
        beta = 0.5
        degradation_model = DegradationModel(alpha, beta)
        return degradation_model


if __name__ == "__main__":
    my_degradation_model = DegradationModel.default()

    from pprint import pprint

    pprint(my_degradation_model)
