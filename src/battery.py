from degradation import DegradationModel
from dataclasses import dataclass

@dataclass()
class Battery:
    def __init__(
        self,
        present_charge,
        degradation_model: DegradationModel,
        DTYPE: type,
        CAPACITY_CHARGE,
        RATE_DISCHARGE,
        RATE_CHARGE,
    ):
        self.present_charge = DTYPE(present_charge)
        self.degradation_model = degradation_model
        self.DTYPE = DTYPE

        self.CAPACITY_CHARGE = DTYPE(CAPACITY_CHARGE)
        self.RATE_DISCHARGE = DTYPE(RATE_DISCHARGE)
        self.RATE_CHARGE = DTYPE(RATE_CHARGE)


    def able_charge(self):
        discharge = min(self.RATE_DISCHARGE, self.present_charge)
        charge = min(self.RATE_CHARGE, self.CAPACITY_CHARGE - self.present_charge)

        return [discharge, charge]

    def charge_discharge(self, charge_amount):
        if not (self.constraint(charge_amount)):
            return
        new_charge_level = self.present_charge + charge_amount
        self.present_charge = new_charge_level

    def constraint(self, charge_amount):
        upper_lower_constraint = bool(
            self.able_discharge() <= charge_amount <= self.able_charge()
        )
        return upper_lower_constraint


if __name__ == "__main__":
    B = 1000
    b_t = 500
    d = 50
    c = 25

    alpha = 0.01
    beta = 0.2

    my_battery = Battery(
        capacity_charge=B,
        present_charge=b_t,
        rate_discharge=d,
        rate_charge=c,
        degradation_model=DegradationModel(alpha, beta),
    )
    a_t = 25

    my_battery.charge_discharge(a_t)

    assert my_battery.present_charge == b_t + a_t
    assert my_battery.able_discharge() == -1 * min(d, b_t)
    assert my_battery.able_charge() == min(c, B - b_t)
