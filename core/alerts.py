import numpy as np
import pandas as pd

def compute_quadrant_alerts(pred_risk_map, ground_truth_map, risk_threshold=0.6, threshold_pixels=150):
    """Compute quadrant alerts with severity levels."""
    rows, cols = pred_risk_map.shape
    mid_row, mid_col = rows // 2, cols // 2
    quadrant_slices = {
        "Top-Left": (slice(mid_row, rows), slice(0, mid_col)),
        "Top-Right": (slice(mid_row, rows), slice(mid_col, cols)),
        "Bottom-Left": (slice(0, mid_row), slice(0, mid_col)),
        "Bottom-Right": (slice(0, mid_row), slice(mid_col, cols)),
    }

    results = []
    for name, (rs, cs) in quadrant_slices.items():
        pr_arr = pred_risk_map[rs, cs]
        gt_arr = ground_truth_map[rs, cs]

        if pr_arr.size == 0:
            results.append({"Quadrant": name, "Pred_Alert": "NO", "Severity": "SAFE"})
            continue

        gt_count = int(np.nansum(gt_arr == 1))
        pixels_over = int(np.nansum(pr_arr >= risk_threshold))
        avg_risk = float(np.nanmean(pr_arr))

        # Severity classification
        if pixels_over > threshold_pixels * 2:
            severity = "CRITICAL"
        elif pixels_over > threshold_pixels:
            severity = "HIGH"
        elif pixels_over > threshold_pixels // 2:
            severity = "MODERATE"
        else:
            severity = "SAFE"

        results.append({
            "Quadrant": name,
            "GT_Pixels": gt_count,
            "Pixels_Over_Thresh": pixels_over,
            "Avg_Risk": round(avg_risk, 3),
            "Pred_Alert": "YES" if severity in ["MODERATE", "HIGH", "CRITICAL"] else "NO",
            "Severity": severity
        })
    return pd.DataFrame(results)


class AlertManager:
    """Keeps track of past alerts to reduce false positives with hysteresis."""
    def __init__(self, persistence=2):
        self.persistence = persistence
        self.history = {}  # {quadrant: consecutive_alerts}

    def update(self, alert_df):
        triggered = []
        for _, row in alert_df.iterrows():
            quad = row["Quadrant"]
            if row["Pred_Alert"] == "YES":
                self.history[quad] = self.history.get(quad, 0) + 1
            else:
                self.history[quad] = 0

            if self.history[quad] >= self.persistence:
                triggered.append(row)

        return pd.DataFrame(triggered)
