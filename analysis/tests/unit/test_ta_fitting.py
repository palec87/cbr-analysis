import numpy as np
import matplotlib.pyplot as plt
from analysis.experiments.ta import Ta
from analysis.modules import fitting as ft


x = (np.linspace(-5, 99, 105),)
par = (1, 5, 1, 50, -1, 200)


def test_single_kin():
    data0 = ft.exp_model(par[:2], x[0], n=1)
    data0 += np.random.normal(0, 0.1, len(x[0]))
    fit = ft.fit_kinetics(x, [data0], n_exp=1)
    plt.plot(x[0], data0)
    plt.show()
    assert list(np.round(fit[0].x)) == [1, 5]
