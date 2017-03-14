import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

def kde_sklearn(x, bandwidth=0.2, **kwargs):
	x_grid = np.linspace(x.min() - 1, x.max() + 1, 500)
	"""Kernel Density Estimation with Scikit-learn"""
	kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
	kde_skl.fit(x[:, np.newaxis])
	# score_samples() returns the log-likelihood of the samples
	log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
	return np.exp(log_pdf), x_grid
	
def MahalanobisDist(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),
                                        inv_covariance_xy),diff_xy[i])))
    return md
    
def FindOutliers(x, y, p):
    MD = MahalanobisDist(x, y)
    nx, ny, outliers = [], [], []
    threshold = -2*np.log(1-p)
    for i in range(len(MD)):
        if MD[i]*MD[i] < threshold:
            nx.append(x[i])
            ny.append(y[i])
            outliers.append(i) # position of removed pair
    return (np.array(nx), np.array(ny), np.array(outliers))
    
   
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

#Add reduced variable for analysis
x = abs(charge_n-payment_n)/charge_n

fig, ax2 = plt.subplots(figsize = (9,7))

#Plot histogram
ax2.hist(x, bins = 100, alpha = 0.5, normed = True)

#Plot KDE
pdf, x_grid = kde_sklearn(x, bandwidth = 0.5)
pdf2, x_grid2 = kde_sklearn(x, bandwidth = 0.1)
pdf3, x_grid3 = kde_sklearn(x, bandwidth = 0.9)
ax2.plot(x_grid, pdf, alpha = 0.9, color = 'green', linewidth = 2.0)
ax2.plot(x_grid2, pdf2, alpha = 0.9, color = 'red', linewidth = 2.0)
ax2.plot(x_grid3, pdf3, alpha = 0.9, color = 'yellow', linewidth = 2.0)
plt.show()

# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.linspace(0, 0.5, 10)}
grid = GridSearchCV(KernelDensity(), params, cv = 20)
grid.fit(x[:, None])

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

xbar = abs(charge_n - allowed_n)/charge

g = sns.jointplot(x, xbar, kind = 'kde', size = 7, space = 0)
plt.show()

md = MahalanobisDist(x,xbar)
Outliers = FindOutliers(x,xbar,0.00000243)

#Print outliers
print "Total Outliers found :", len(Outliers[2])
print "The index of the variables are :", Outliers[2]
