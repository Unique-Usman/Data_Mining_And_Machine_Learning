"""
A Python script to that uses K-Means algorithm to find the clusters
From a generates random data. 
It uses the elbow method to find the optimal number of clusters
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def generate_synthetic_data():
    """
    Generate synthetic data using np.random.seed to
    ensure consistency during the generation of the data. 
    Plot the value of the data also. 
    """

    np.random.seed(0)
    size = 1000
    mu1 = [2, 2]
    sigma1 = [[0.9, -0.0255], [-0.0255, 0.9]]
    data_point1 = np.random.multivariate_normal(mu1, sigma1, size)

    mu2 = [5, 5]
    sigma2 = [[0.5, 0], [0, 0.3]]
    data_point2 = np.random.multivariate_normal(mu2, sigma2, size)

    mu3 = [-2, -2]
    sigma3 = [[1, 0], [0, 0.9]]
    data_point3 = np.random.multivariate_normal(mu3, sigma3, size)

    mu4 = [-4, 8]
    sigma4 = [[0.8, 0], [0, 0.6]]
    data_point4 = np.random.multivariate_normal(mu4, sigma4, size)

    combined_data_point = np.vstack([data_point1, data_point2,
                                     data_point3, data_point4])

    # Shuffle the data to ensure randomness
    np.random.shuffle(combined_data_point)

    plt.scatter(combined_data_point[:, 0], combined_data_point[:, 1], s=10, alpha=0.7)
    plt.title('Random Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    return combined_data_point

def k_value_elbow(data):
    """
    Finding the optimal k value and also plotting the Elbow method for
    Better visualization.
    n_init is default to 10 but,it was giving warning when I did not indicate it.
    """
    wcss=[]
    for i in range(1,21):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
        kmeans.fit(data)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    plt.plot(range(1, 21), wcss, marker='o')
    plt.title('Elbow Method for Optimal Cluster Count')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()

def perform_kmeans(data):
    """
    Perform k-means clustering with optimal number of
    clusters that is given by the elbow method
    """
    number_of_k = 4
    Kmeans = KMeans(n_clusters=number_of_k, random_state=0, n_init=10)
    Kmeans.fit(data)
    cluster_assignments = Kmeans.labels_
    for i in range(number_of_k):
        cluster_data = data[cluster_assignments == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, alpha=0.7, label=f'Cluster {i + 1}')

    # Adding labels and title
    plt.title('K-Means Clustering - Clustered Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Adding a legend for each cluster
    plt.legend(title='Clusters', loc='upper right')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    data = generate_synthetic_data()
    k_value_elbow(data)
    perform_kmeans(data)
