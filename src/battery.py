from .degradation import DegradationModel
from dataclasses import dataclass
from .environment_typing import envt


@dataclass
class Battery:
    present_charge: envt.Any
    CAPACITY_CHARGE: envt.Any
    RATE_DISCHARGE: envt.Any
    RATE_CHARGE: envt.Any
    degradation_model: DegradationModel

    def __post_init__(
        self,
    ):
        self.present_charge = envt.uint(self.present_charge)
        self.CAPACITY_CHARGE = envt.uint(self.CAPACITY_CHARGE)
        self.RATE_DISCHARGE = envt.int(self.RATE_DISCHARGE)  # not uint because -1*D
        self.RATE_CHARGE = envt.int(self.RATE_CHARGE)

    def able_charge(self):
        discharge = min(self.RATE_DISCHARGE, self.present_charge)
        charge = min(self.RATE_CHARGE, self.CAPACITY_CHARGE - self.present_charge)

        return [discharge, charge]

    def charge_discharge(self, charge_amount):
        assert self.constraint(charge_amount)
        new_charge_level = self.present_charge + charge_amount
        self.present_charge = new_charge_level

    def constraint(self, charge_amount):
        D, C = self.able_charge()
        upper_lower_constraint = bool(-D <= charge_amount <= C)
        return upper_lower_constraint


if __name__ == "__main__":
    B = 1000
    b_t = 500
    d = 50
    c = 25

    alpha = envt.float(0.01)
    beta = envt.float(0.2)

    my_battery = Battery(
        present_charge=b_t,
        CAPACITY_CHARGE=B,
        RATE_DISCHARGE=d,
        RATE_CHARGE=c,
        degradation_model=DegradationModel(alpha, beta),
    )

    a_t = 25
    my_battery.charge_discharge(a_t)
    assert my_battery.present_charge == b_t + a_t

    D, C = my_battery.able_charge()
    assert [-D, C] == [-min(d, b_t), min(c, B - b_t)]

    from pprint import pprint

    pprint(my_battery)
