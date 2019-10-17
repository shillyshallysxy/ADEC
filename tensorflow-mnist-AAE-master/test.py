# -*- encoding:utf8 -*-
from plot_utils import Plot_Manifold_Learning_Result
z = Plot_Manifold_Learning_Result("111").z
import matplotlib.pyplot as plt
plt.scatter(z[:,0], z[:,1])
plt.show()
print(1)