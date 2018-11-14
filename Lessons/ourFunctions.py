import numpy as np
import matplotlib as matplt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy import stats

from numpy import random 
#random.seed(42)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

def bs_2sample_test(xA, xB, func, direction =("two-sided","left", "right")[0], size=1000):
    # Compute "pooled" mean 
    mean_overall = np.mean([xA,xB])
    
    empirical_diff_means = np.mean(xA)-np.mean(xB)

    # Generate shifted arrays
    xA_underNull = xA - np.mean(xA) + mean_overall
    xB_underNull = xB - np.mean(xB) + mean_overall

    # Compute 10,000 bootstrap replicates from shifted arrays
    bs_replicates_m = draw_bs_reps(xA_underNull, np.mean, size=size)
    bs_replicates_f = draw_bs_reps(xB_underNull, np.mean, size=size)

    # Get replicates of difference of means: bs_replicates
    bs_replicates = bs_replicates_m - bs_replicates_f

    # Compute and print p-value: p
    if direction == "two-sided":        
        p = np.sum(np.abs(bs_replicates) >= np.abs(empirical_diff_means)) / len(bs_replicates)
    if direction == "left":        
        p = np.sum(bs_replicates <= empirical_diff_means) / len(bs_replicates)
    if direction == "right":        
        p = np.sum(bs_replicates >= empirical_diff_means) / len(bs_replicates)
    print('p-value =', p)
    
    return bs_replicates

def plot2ECDFs(x1, x2,leg=('male', 'female'),xlab='birth weight(g)',ylab='ECDF',title=''):
    # Compute ECDF for sample size 40: m_40, f_40
    mx_40, my_40 = ecdf(x1)
    fx_40, fy_40 = ecdf(x2)

    # Plot all ECDFs on the same plot
    _ = plt.plot(mx_40, my_40, marker = '.', linestyle = 'none')
    _ = plt.plot(fx_40, fy_40, marker = '.', linestyle = 'none')

    # Make nice margins
    plt.margins(0.02)

    # Annotate the plot
    plt.legend(leg, loc='lower right')
    _ = plt.xlabel(xlab)
    _ = plt.ylabel(ylab)
    _ = plt.title(title)

    # Display the plot
    plt.grid()
    plt.show()

def mean_density_comparison(M=500, n=10):
    
    #Generate an gender iteration array
    gender_iter = ['male', 'female']
    
    #Create an empty DataFrame with 'gender' and 'dbirwt' column
    columns = ['gender', 'dbirwt']
    df_new = pd.DataFrame(columns=columns)
    
    #Create an empty array to store the standard deviation of the differnt gender 'male' = std_dev[0], 'female' = std_dev[1]
    std_dev = np.empty(2)
    
    #Iterate over gender and create a specific data subset
    for ind,v in enumerate(gender_iter):
        subset = df_cleaned[df_cleaned.gender == v]
        
        #create M random sample means of n samples and add it to df_new
        for i in range(M):
            rand_samples = np.random.choice(subset.dbirwt, n)
            x = np.mean(rand_samples)
            df_new.loc[len(df_new)+1] = [v, x]
        
        #plot male and female data and calculate the standard daviation of the data
        plot_data = df_new[df_new.gender == v]
        std_dev[ind] = np.std(plot_data['dbirwt'])  
        
        plot_data.dbirwt.plot.density()
        plt.xlabel('dbirwt')
        plt.legend(gender_iter)
		#plt.grid()
		#plt.title("n=" + str(n))
		
    #return the sample mean data
    return df_new
    #return the standard deviation of ['male', 'female']
    #return std_dev
        
#Test the function
#SM40 = mean_density_comparison(M=100, n=40)
#plt.figure()
#SM640 = mean_density_comparison(M=100, n=640)

## Permutation tests

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff