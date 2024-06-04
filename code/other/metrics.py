from scipy.spatial import distance
import numpy as np

def count_clustering_scores(X, cluster_num, model_instance, score_fun):
    if isinstance(cluster_num, int):
        cluster_num_iter = [cluster_num]
    else:
        cluster_num_iter = cluster_num
        
    scores = []    
    for k in cluster_num_iter:
        labels = model_instance.fit_predict(X)
        wcss = score_fun(X, labels)
        scores.append(wcss)
    
    if isinstance(cluster_num, int):
        return scores[0]
    else:
        return scores 

############################################

def min_interclust_dist(X, label):
    clusters = set(label)
    global_min_dist = np.inf
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)[0].tolist()
        for cluster_j in clusters:
            if cluster_i != cluster_j:
                cluster_j_idx = np.where(label == cluster_j)[0].tolist()
                interclust_min_dist = np.min(distance.cdist(X.iloc[cluster_i_idx], X.iloc[cluster_j_idx]))
                global_min_dist = np.min([global_min_dist, interclust_min_dist])
    return global_min_dist

def _inclust_mean_dists(X, label):
    clusters = set(label)
    inclust_dist_list = []
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)[0].tolist()
        inclust_dist = np.mean(distance.pdist(X.iloc[cluster_i_idx]))
        inclust_dist_list.append(inclust_dist)
    return inclust_dist_list

def mean_inclust_dist(X, label):
    inclust_dist_list = _inclust_mean_dists(X, label)
    return np.mean(inclust_dist_list)

def std_dev_of_inclust_dist(X, label):
    inclust_dist_list = _inclust_mean_dists(X, label)
    return np.std(inclust_dist_list)

def mean_dist_to_center(X, label):
    clusters = set(label)
    inclust_dist_list = []
    for cluster_i in clusters:
        cluster_i_idx = np.where(label == cluster_i)[0].tolist()
        cluster_i_mean = np.mean(X.iloc[cluster_i_idx], axis=0)
        inclust_dist = np.mean(distance.cdist(X.iloc[cluster_i_idx], cluster_i_mean))
        inclust_dist_list.append(inclust_dist)
    return np.mean(inclust_dist_list)

def print_metrics(X, cluster_num, model_instance):
    print(f'Minimal distance between clusters = '
          f'{count_clustering_scores(X, cluster_num, model_instance, min_interclust_dist):.2f}.')
    print(f'Average distance between points in the same class = '
          f'{count_clustering_scores(X, cluster_num, model_instance, mean_inclust_dist):.2f}.')
    print(f'Standard deviation of distance between points in the same class = '
            f'{count_clustering_scores(X, cluster_num, model_instance, std_dev_of_inclust_dist):.3f}.')
    # print(f'Average distance to cluster center = '
    #         f'{count_clustering_scores(X, cluster_num, model_instance, mean_dist_to_center):.2f}.')

    # chuj nie działa ta czwarta a nie chce mi się teraz 2h spędzić nad tym bo się trzeba na kolosy uczyć xd