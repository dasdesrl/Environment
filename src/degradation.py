from dataclasses import dataclass
from numpy import exp


@dataclass
class DegradationModel:
    alpha: float
    beta: float

    def percent_degradation(self, depth_of_discharge):
        stress_function_degradation = self.alpha * exp(-self.beta * depth_of_discharge)
        return stress_function_degradation
