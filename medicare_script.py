import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

file = 'Medicare_Data_CA_2012.csv'
ca_data = pd.read_csv(file)

#Create figure 
f, ax = plt.subplots(figsize = (9,7))

#Check corelation between different variables 
corplot = ca_data.corr()

#Draw heatmap 
plot = sns.heatmap(corplot, vmin=corplot.values.min(), vmax=1, square=True, 
        linewidths=0.1, annot=True, annot_kws={"size":8})

for label in plot.get_yticklabels():
        label.set_size(8)
        label.set_weight("bold")
        label.set_rotation(0)
        
for label in plot.get_xticklabels():
        label.set_size(8)
        label.set_weight("bold")
        label.set_rotation(90)

plt.show()

payment = ca_data['average_Medicare_payment_amt']
charge = ca_data['average_submitted_chrg_amt']
allowed_amount = ca_data['average_Medicare_allowed_amt']

payment_n = payment.values
charge_n = charge.values
allowed_n = allowed_amount.values

f0 = charge_n
f1 = payment_n
f2 = allowed_n 

x = abs(charge_n-payment_n)/charge_n




#grid = GridSearchCV(KernelDensity(),
#                    {'bandwidth': np.linspace(0.1, 1.0, 5)},
#                    cv=20)
#grid.fit(payment_n[:, np.newaxis])
#print grid.best_params_

#x_grid = np.linspace(0, 600, 100)
#fig, ax = plt.subplots()
#ax.hist(payment_n, bins = 100, range = (0, 600), normed = True, histtype = 'stepfilled', alpha = 0.5)
#plt.hist(charge_n, bins = 100, range = (0, 600), normed = True, histtype = 'stepfilled', color = 'yellow')

#pdf_1 = kde_sklearn(payment_n, x_grid, bandwidth = 0.1)
#pdf_3 = kde_sklearn(payment_n, x_grid, bandwidth = 0.9)

#ax.plot(x_grid, pdf_1, alpha = 0.5, color = 'blue')
#ax.plot(x_grid, pdf_3, alpha = 0.5, color = 'green')


#plt.show()