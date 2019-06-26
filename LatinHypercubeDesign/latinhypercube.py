

from pyDOE import *
from scipy.stats.distributions import norm
 

x = lhs(4, samples=40, criterion='center')

#print(x)

design = lhs(4, samples=10)
means = [1.5, 0.0005, 3, 4]
stdvs = [0.5, 0.00005, 1, 0.25]
for i in xrange(4):
	design[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(design[:, i])

print(design)