# core/confidence.py
import numpy as np

def prediction_confidence(preds, n_bootstrap=100):
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(preds, size=len(preds), replace=True)
        boot_means.append(sample.mean())
    mean = np.mean(boot_means)
    lower = np.percentile(boot_means, 5)
    upper = np.percentile(boot_means, 95)
    return mean, (lower, upper)
