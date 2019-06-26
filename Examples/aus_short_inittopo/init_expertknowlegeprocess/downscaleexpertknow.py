
 
import scipy.ndimage as ndimage  
from scipy import stats  

import numpy as np
import seaborn as sns; sns.set()

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')


expert_know = np.loadtxt('init_estimated.txt')
 
 
ax = sns.heatmap(expert_know) 
fig = ax.get_figure()
fig.savefig("output.png")
fig.clear()

expert_rotate = expert_know.T


ax = sns.heatmap(expert_rotate) 
fig = ax.get_figure()
fig.savefig("output_.png")

fig.clear()

expert_rotate2 = expert_rotate.T


ax = sns.heatmap(expert_rotate2) 
fig = ax.get_figure()
fig.savefig("output_x.png")

fig.clear()


expert_rotate3 = expert_rotate2.T


ax = sns.heatmap(expert_rotate3) 
fig = ax.get_figure()
fig.savefig("output_x2.png")

fig.clear()