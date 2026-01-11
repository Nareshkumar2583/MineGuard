import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import time
import rasterio
from scipy.ndimage import label
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestRegressor

# 🔑 Modular imports
from core.confidence import prediction_confidence
from core.forecasting import forecast_next
from core.logs import save_event

# -----------------------
# Config / paths
# -----------------------
DEM_PATH = r"data/synthetic_pit_dem.tif"
MODEL_PATH = r"data/rf_regressor.pkl"

risk_threshold = 0.6
threshold_pixels = 150
min_cluster_size = 50
batch_interval = 2   # seconds
batch_size = 500

# -----------------------
# Email Settings
# -----------------------
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", "").split(",")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))


def send_email_alert(alert_df):
    subject = "🚨 Mine Risk Alert Triggered!"
    body_html = f"""
    <html><body>
        <h2 style="color:red;">🚨 Mine Monitoring Alert</h2>
        <p>The system detected one or more <b>high-risk quadrants</b>.</p>
        <h3>Alert Summary:</h3>
        {alert_df.to_html(index=False, border=1)}
    </body></html>
    """
    try:
        msg = MIMEText(body_html, "html")
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = ", ".join(ALERT_RECEIVER)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, ALERT_RECEIVER, msg.as_string())
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)


# -----------------------
# DEM + Model
# -----------------------
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1)

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)


# -----------------------
# Helpers
# -----------------------
def clean_clusters(binary_map, min_cluster_size=50):
    labeled_array, num_features = label(binary_map)
    clean_map = np.zeros_like(binary_map)
    for i in range(1, num_features + 1):
        cluster = (labeled_array == i)
        if cluster.sum() >= min_cluster_size:
            clean_map[cluster] = 1
    return clean_map


def weighted_risk_map(pred_risk_map, threshold=0.6):
    return np.where(pred_risk_map >= threshold, pred_risk_map ** 2, pred_risk_map * 0.5)


class AlertManager:
    def __init__(self, persistence=2, history_len=8):
        self.persistence = persistence
        self.history_len = history_len
        self.history = {}

    def update(self, alert_df):
        final_alerts = []
        for _, row in alert_df.iterrows():
            q = row["Quadrant"]
            risky_pixels = row[f"Pred_Pixels>={risk_threshold:.2f}"]
            if q not in self.history:
                self.history[q] = []
            self.history[q].append(risky_pixels)
            if len(self.history[q]) > self.history_len:
                self.history[q].pop(0)

            if risky_pixels >= 300:
                severity = "🔴 Critical"
            elif risky_pixels >= 150:
                severity = "🚨 High"
            elif risky_pixels >= 100:
                severity = "⚠️ Low"
            else:
                severity = "✅ Safe"

            trend = "Stable"
            if len(self.history[q]) >= 3:
                slope = np.polyfit(range(len(self.history[q])), self.history[q], 1)[0]
                if slope > 0: trend = "Rising"
                elif slope < 0: trend = "Falling"

            pred_alert = row["Pred_Alert"]
            if pred_alert == "YES":
                last_n = self.history[q][-self.persistence:]
                if all(p >= threshold_pixels for p in last_n):
                    final_alerts.append({
                        "Quadrant": q, "Pixels": risky_pixels,
                        "Severity": severity, "Trend": trend
                    })
        return pd.DataFrame(final_alerts)


def compute_quadrant_alerts(pred_risk_map, ground_truth_map):
    rows, cols = pred_risk_map.shape
    mid_row, mid_col = rows // 2, cols // 2
    slices = {
        "Top-Left": (slice(mid_row, rows), slice(0, mid_col)),
        "Top-Right": (slice(mid_row, rows), slice(mid_col, cols)),
        "Bottom-Left": (slice(0, mid_row), slice(0, mid_col)),
        "Bottom-Right": (slice(0, mid_row), slice(mid_col, cols)),
    }
    results = []
    for name, (rs, cs) in slices.items():
        gt_arr = ground_truth_map[rs, cs]
        pr_arr = pred_risk_map[rs, cs]
        if gt_arr.size == 0 or pr_arr.size == 0:
            results.append({"Quadrant": name, "Pred_Alert": "NO", "GT_Alert": "NO"})
            continue
        gt_count = int(np.nansum(gt_arr == 1))
        avg_risk = float(np.nanmean(pr_arr))
        max_risk = float(np.nanmax(pr_arr))
        pixels_over_thresh = int(np.nansum(pr_arr >= risk_threshold))
        pred_alert = pixels_over_thresh > threshold_pixels
        gt_alert = gt_count > threshold_pixels
        results.append({
            "Quadrant": name,
            "GT_Risky_Pixels": gt_count,
            f"Pred_Pixels>={risk_threshold:.2f}": pixels_over_thresh,
            "Pred_Avg": round(avg_risk, 3),
            "Pred_Max": round(max_risk, 3),
            "Pred_Alert": "YES" if pred_alert else "NO",
            "GT_Alert": "YES" if gt_alert else "NO"
        })
    return pd.DataFrame(results)


# -----------------------
# Streamlit App
# -----------------------
st.title("⛏️ Mine Monitoring Dashboard (Infinite Loop Mode)")

uploaded_file = st.file_uploader("📂 Upload a dataset (CSV)", type=["csv"])

stop_flag = st.checkbox("🛑 Stop after current cycle")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Risk_Score" not in df:
        np.random.seed(42)
        df["Risk_Score"] = df["Risk_Label"].apply(
            lambda x: np.random.uniform(0.7, 1.0) if x == 1 else np.random.uniform(0.0, 0.3)
        )

    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    if "cycle" not in st.session_state:
        st.session_state.cycle = 1

    placeholder = st.empty()

    # 🚀 Infinite loop
    while True:
        st.write(f"🔁 Current Cycle: {st.session_state.cycle}")

        # Train model on current cycle
        X_train = df.drop(columns=["Risk_Label", "Risk_Score", "X", "Y"])
        y_train = df["Risk_Score"]
        model.fit(X_train, y_train)

        num_batches = (len(df) // batch_size) + 1
        for i in range(num_batches):
            batch_df = df.iloc[: (i + 1) * batch_size].copy()
            X = batch_df.drop(columns=["Risk_Label", "Risk_Score", "X", "Y"])
            batch_df["Predicted_Risk"] = model.predict(X)

            # Maps
            ground_truth_map = batch_df.pivot_table(index="Y", columns="X", values="Risk_Label").values
            pred_risk_map = batch_df.pivot_table(index="Y", columns="X", values="Predicted_Risk").values
            binary_map = (pred_risk_map >= risk_threshold).astype(int)
            clean_risk_map = clean_clusters(binary_map, min_cluster_size)
            weighted_map = weighted_risk_map(pred_risk_map, risk_threshold)

            with placeholder.container():
                st.subheader(f"Cycle {st.session_state.cycle} — Batch {i+1}/{num_batches}")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(dem, cmap="terrain", origin="lower")
                ax.imshow(weighted_map, cmap="hot", alpha=0.25, origin="lower", vmin=0, vmax=1)
                ax.contour(clean_risk_map, levels=[0.5], colors="red", linewidths=1.5, origin="lower")
                ax.set_title("Predicted Weighted Risk Zones (Streaming)")
                ax.axis("off")
                st.pyplot(fig)

                alert_df = compute_quadrant_alerts(pred_risk_map, ground_truth_map)
                st.dataframe(alert_df)

                # Process alerts
                final_alerts = st.session_state.alert_manager.update(alert_df)
                if not final_alerts.empty:
                    st.error("🚨 ALERT: Persistent Risk detected!")
                    st.dataframe(final_alerts)
                    ok, msg = send_email_alert(final_alerts)
                    if ok: st.success("✅ Email sent successfully")
                    else: st.warning(f"⚠️ Email failed: {msg}")
                else:
                    st.success("✅ All quadrants safe.")

            if i < num_batches - 1:
                time.sleep(batch_interval)

        # 🔄 End of cycle → feed predictions as next cycle's input
        df["Risk_Score"] = model.predict(df.drop(columns=["Risk_Label", "Risk_Score", "X", "Y"]))
        df.to_csv(f"data/cycle_{st.session_state.cycle}_predictions.csv", index=False)
        st.session_state.cycle += 1
        st.success("✅ Cycle complete — feeding predictions into next cycle.")

        # ✅ Break condition
        if stop_flag:
            st.warning("🛑 Loop stopped after this cycle.")
            break
