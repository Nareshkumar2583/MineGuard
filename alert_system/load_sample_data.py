import pandas as pd
from src.db import save_alert_df  # or save_alert_rows

# load your dataset
df = pd.read_csv("data/quadrant_alerts_predicted_and_gt.csv")

# save snapshot (entire dataframe as one document)
inserted_id = save_alert_df(df, meta={"source": "initial_test"})
print(f"✅ Inserted sample alert snapshot with id={inserted_id}")
