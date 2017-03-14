# outlier_kde
Using kernel density estimation to detect outliers in California's medicare data

Medicare in US is a health insurance program for people above the age of 65 in USA. The dataset is publically available on the internet. I thought this will be an interesting unsupervised machine learning problem. 
The question I was probing was - Are their any outliers in the medicare program who demonstrate a different charged, paid and availed amount for the program. 

I initially started with plotting the data on a histogram and check for covariance in the dataset. The dataset is huge, so the code will run slow. 

It demonstrated that there is a strong covariance between the charged and allowed amount for the program. The corelation is of around 0.999, with a p value of 0, clearnly demonstrating a linear relationship. 

So, I reduced the dimensionality to one by considering the independent variable as: x = abs(charge_n-payment_n)/charge_n

The next step was to use a gaussian kernel to smoothen the histogram, followed by a Grid search cross validation to optimize the bandwidth. 

Since, it was a simple case of one variable, I used Mahalanobis distance to find the outliers. The distance is the linear distance from the expected value of the kernel. 

Result - 4 outliers with 5 sigma confidence level. 
