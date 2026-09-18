from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "transactions.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["order_date"])
snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("customer_id")
      .agg(
          recency=("order_date", lambda x: (snapshot_date - x.max()).days),
          frequency=("order_id", "nunique"),
          monetary=("revenue", "sum"),
      )
)

for col in ["recency", "frequency", "monetary"]:
    rfm[f"{col}_score"] = pd.qcut(
        rfm[col].rank(method="first"), 5, labels=False
    ) + 1

# Lower recency is better; higher frequency/monetary is better.
rfm["r_score"] = 6 - rfm["recency_score"]
rfm["f_score"] = rfm["frequency_score"]
rfm["m_score"] = rfm["monetary_score"]

def segment(row):
    if row.r_score >= 4 and row.f_score >= 4 and row.m_score >= 4:
        return "Champions"
    if row.r_score >= 4 and row.f_score >= 3:
        return "Loyal Customers"
    if row.r_score <= 2 and row.m_score >= 3:
        return "At Risk - High Value"
    if row.r_score <= 2:
        return "Needs Reactivation"
    return "Potential Loyalists"

rfm["segment"] = rfm.apply(segment, axis=1)
rfm.to_csv(OUT / "customer_segments.csv")

summary = (
    rfm.groupby("segment")
       .agg(customers=("segment", "size"), revenue=("monetary", "sum"))
       .sort_values("revenue", ascending=False)
)

print(summary)
