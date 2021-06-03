import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import ttest_ind

p_values_hm = [0.1667, 0.4347, 0.0025, 0.4754] #p-values from Kate from human-macaque pairs
p_values_us = [0.2457, 0.0054, 0.1352, 0.0882] #p-values from Kate from untreated-stained pairs

def fisher_p(p_values):
	"""Takes list of p-values and returns the test statistic according to Fisher's combined probability test"""
	return -2*np.sum(np.log(p_values))

def extreme_probability(p_values, plot = False):
	s = fisher_p(p_values)
	n = len(p_values)
	p_extreme = chi2.sf(s, 2*n)
	if plot:
		x = np.linspace(0, chi2.isf(1E-5, 2*n), 1000)
		plt.axvline(s, c = 'r')
		plt.plot(x, chi2.pdf(x, 2*n))
	return p_extreme

plt.figure()
plt.title("Combined p-value for human-macaque pairs")
p_extreme_hm = extreme_probability(p_values_hm, plot = True)
print("Probability of combined p-value human-macaque pairs = {:.2f}%".format(p_extreme_hm*100))

plt.figure()
plt.title("Combined p-value for untreated-stained pairs")
p_extreme_us = extreme_probability(p_values_us, plot = True)
print("Probability of combined p-value human-macaque pairs = {:.2f}%".format(p_extreme_us*100))

plt.figure()
plt.title("Distribution of combined p values under null hypothesis")
n = 4
s_values = [fisher_p([ttest_ind(np.random.randn(100), np.random.randn(100))[1] for pair in range(n)]) for trial in range(1000)]
plt.hist(s_values, density = True, histtype = 'stepfilled')
x = np.linspace(0, chi2.isf(1E-5, 2*n), 1000)
plt.plot(x, chi2.pdf(x, 2*n))
plt.show()


