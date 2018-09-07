import cntk as C
from learner.clr import CyclicalLeaningRate
import matplotlib.pyplot as plt


a = C.input_variable(10)
model = C.layers.Dense(10)(a)
sgd = C.sgd(model.parameters, 0.01)
clr = CyclicalLeaningRate(sgd, lr_policy="exp_range", max_lrs=20)
lr_schedule = clr.get_lr_schedule()

plt.scatter(range(lr_schedule.shape[0]), lr_schedule[:, 0], s=1)
plt.show()
