# src/notify.py
import os
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

def _normalize_columns(df: pd.DataFrame):
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    return df

def send_email(message: str):
    receivers = [email.strip() for email in ALERT_RECEIVER.split(",") if email.strip()]
    if not receivers:
        return False, "No recipients configured"

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(receivers)
    msg["Subject"] = "⚠️ Mine Rockfall Risk Alert"
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, receivers, msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)

def send_alert_from_dataframe(df: pd.DataFrame):
    """
    df: DataFrame with columns at least:
      - Quadrant (name)
      - Pred_Pixels... (predicted pixel count column)
      - Pred_Alert (YES/NO or True/False)
    Returns (ok:bool, message_or_error)
    """
    df = _normalize_columns(df)

    # try to find needed columns robustly
    quadrant_col = next((c for c in df.columns if "quad" in c.lower()), None)
    pred_alert_col = next((c for c in df.columns if "pred_alert" in c.lower()), None)
    pred_pixels_col = next((c for c in df.columns if "pred_pixels" in c.lower() or "pred_pixels" in c.lower().replace(">", "_")), None)

    # fallback to patterns
    if pred_pixels_col is None:
        # find any column with 'Pred' and numeric values
        for c in df.columns:
            if "pred" in c.lower() and df[c].dtype.kind in "iuf":
                pred_pixels_col = c
                break

    if not quadrant_col:
        return False, "Quadrant column not found"
    if not pred_alert_col:
        return False, "Pred_Alert column not found"
    if not pred_pixels_col:
        return False, "Predicted pixels column not found"

    # Normalize Pred_Alert values to boolean
    def is_yes(x):
        if pd.isna(x): return False
        if isinstance(x, bool): return x
        s = str(x).strip().lower()
        return s in ("yes", "y", "true", "1")

    df["_pred_alert_bool"] = df[pred_alert_col].apply(is_yes)

    risky = df[df["_pred_alert_bool"]]
    if risky.empty:
        # still send summary? you can choose not to send if no risky quadrants
        return False, "No predicted alerts (no Pred_Alert == YES)"

    # first risky quadrant
    first = risky.iloc[0]
    first_quad = first[quadrant_col]
    first_pixels = int(first[pred_pixels_col])

    # Build message text
    message = f"⚠️ First detected risk in {first_quad} with {first_pixels} risky pixels.\n\n"
    message += "Quadrant-wise rockfall prediction and predicted risky pixels:\n\n"

    # iterate all quadrants in original df (so we show full status)
    for _, row in df.iterrows():
        quad = row[quadrant_col]
        pa = "YES" if is_yes(row[pred_alert_col]) else "NO"
        pixels = int(row[pred_pixels_col]) if pd.notna(row[pred_pixels_col]) else 0
        message += f"- {quad}: Rockfall={pa}, Risky Pixels={pixels}\n"

    # send email
    ok, err = send_email(message)
    if ok:
        return True, None
    else:
        return False, err
