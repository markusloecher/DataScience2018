'''
Bootstrap replicates of the mean and the SEM

import matplotlib.pyplot as plt
import numpy as np

def bootstrap_replicate_1d(data):
    return np.mean(np.random.choice(data, size=len(data)))
	
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(, np.mean, size=10000)

# Compute and print SEM
sem = np.std() / np.sqrt(len())
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))