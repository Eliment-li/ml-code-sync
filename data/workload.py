import matplotlib.pyplot as plt
import numpy as np


def gauss_distribution(mu,sigma):

    s = np.random.normal(mu, sigma, 1000)

    # Create the bins and histogram
    count, bins, ignored = plt.hist(s, 20)

    # Plot the distribution curve
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),       linewidth=3, color='y')
    plt.show()


if __name__ == '__main__':
    gauss_distribution(0.5,0.05)
    gauss_distribution(0.1,0.05)
    gauss_distribution(1,0.05)


