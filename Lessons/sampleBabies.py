import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import random 
random.seed(42)

df = pd.read_csv('../data/BirthWeights.csv')
#Remove weights below 500 and above 8000
df_cleaned = df[np.logical_and(df.dbirwt > 500, df.dbirwt < 8000)]

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
    #return the standard daeviation of ['male', 'female']
    #return std_dev
        
#Test the function
SM40 = mean_density_comparison(M=100, n=40)
plt.figure()
SM640 = mean_density_comparison(M=100, n=640)


SM40.head()
grouped = SM40["dbirwt"].groupby(SM40["gender"])
print(grouped.mean())
print(grouped.std())