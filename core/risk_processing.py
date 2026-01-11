import numpy as np
from scipy.ndimage import label

def clean_clusters(binary_map, min_cluster_size=50):
    """Remove small noisy clusters from risk binary map."""
    labeled_array, num_features = label(binary_map)
    clean_map = np.zeros_like(binary_map)
    for i in range(1, num_features + 1):
        cluster = (labeled_array == i)
        if cluster.sum() >= min_cluster_size:
            clean_map[cluster] = 1
    return clean_map

def weighted_risk_map(pred_risk_map, risk_threshold=0.6):
    """Generate weighted risk heatmap instead of strict binary."""
    norm_map = np.clip((pred_risk_map - risk_threshold) / (1 - risk_threshold), 0, 1)
    return norm_map
