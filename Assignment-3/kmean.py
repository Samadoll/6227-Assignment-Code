import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import os

_true_centroids = []

def get_random_point(n, min_distance):
    np.random.seed(1333)
    points = []
    while len(points) < n:
        point = np.random.rand(2)
        if all(np.linalg.norm(point - existing_point) >= min_distance for existing_point in points):
            points.append(point)
    return points


def get_initial_points():
    initial_points = [
        [0, 0],
        [30, 0],
        [0, 40],
        [100, 0],
        [50, 50]
    ]
    return initial_points


def get_random_within_r(point, radius):
    points = []
    for i in range(20):
        angle = random.uniform(0, 2*math.pi)
        distance = random.uniform(0, radius)
        points.append([point[0] + distance * math.cos(angle), point[1] + distance * math.sin(angle)])
    return points


def plot_points(points, labels, centers):
    plt.scatter([x[0] for x in points], [x[1] for x in points], c=labels)
    plt.scatter([x[0] for x in centers], [x[1] for x in centers], marker='*', s=200, linewidth=3, color='r')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def kmean(points):
    np.random.seed(0)
    X = np.random.randn(100, 2)
    print(type(X))
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(points)

    # Get the cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    plot_points(points, labels, centers)
    # Plot the data points and cluster centers
    # plt.scatter(X[:, 0], X[:, 1], c=labels)
    # plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, linewidth=3, color='r')
    # plt.show()


def kmean_n_step(points):
    data = np.array(points)
    centroids = np.array([
        [0, 10],
        [20, 10],
        [0, 30],
        [40, 10],
        [10, 30]
    ])
    centroids_b = np.array([
        [95, -5],
        [95, 5],
        [105, -5],
        [105, 5],
        [100, 0]
    ])
    t = round(time.time())
    os.makedirs(str(t))
    plot_kmean_n_step(str(t), data, centroids, "A")
    plot_kmean_n_step(str(t), data, centroids_b, "B")


def plot_kmean_n_step(folder, data, centroids, success):
    plt.scatter(data[:,0], data[:,1], color='black')
    plt.scatter(centroids[:,0], centroids[:,1], color='red')
    true_centroid_pts = np.array(_true_centroids)

    # Iterate through the K-means algorithm
    for i in range(5):
        # Assign each data point to the closest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Plot the data points and centroids
        plt.figure()
        plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis')
        plt.scatter(true_centroid_pts[:,0], true_centroid_pts[:,1], marker="^", color='orange', s=150)
        plt.scatter(centroids[:,0], centroids[:,1], color='red')
        plt.savefig(f"{folder}/{success}_{i}_1.png")
        
        # Update the centroids based on the new cluster assignments
        centroids = np.array([data[labels == j].mean(axis=0) for j in range(len(centroids))])
        
        # Plot the updated centroids
        plt.figure()
        plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis')
        plt.scatter(centroids[:,0], centroids[:,1], color='red')
        plt.savefig(f"{folder}/{success}_{i}_2.png")

    # if success == "B":
    #     print(centroids)
    
    plt.show()



def run():
    initial_points = get_initial_points()
    all_points = []
    for point in initial_points:
        random_pts = get_random_within_r(point, 10)
        all_points += random_pts
        set_true_centroid([point] + random_pts)
    all_points += initial_points
    # plot_points(all_points)
    kmean_n_step(all_points)


def set_true_centroid(data):
    _true_centroids.append(np.mean(np.array(data), axis=0).tolist())

run()

