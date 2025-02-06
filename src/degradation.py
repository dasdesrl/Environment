import numpy as np
from dataclasses import dataclass, InitVar

DTYPE_FLOAT = np.float32

@dataclass
class DegradationModel:
    alpha: InitVar[DTYPE_FLOAT]
    beta: InitVar[DTYPE_FLOAT]

    def __post_init__(self, alpha, beta):
        self.__alpha = alpha
        self.__beta = beta
        
    def percent_degradation(self, depth_of_discharge):
        stress_function_degradation = self.__alpha * np.exp(-self.__beta * depth_of_discharge)
        return stress_function_degradation
