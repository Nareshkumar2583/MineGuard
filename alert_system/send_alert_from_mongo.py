# alert_system/send_alert_from_mongo.py
from src.db import get_latest_alert_df
from src.notify import send_alert_from_dataframe

def main():
    df, doc = get_latest_alert_df()
    if df is None:
        print("No alert documents found in MongoDB.")
        return

    ok, err = send_alert_from_dataframe(df)
    if ok:
        print("Alert sent successfully.")
    else:
        print("Failed to send alert:", err)

if __name__ == "__main__":
    main()
