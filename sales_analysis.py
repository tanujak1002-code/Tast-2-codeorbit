import numpy as np
import pandas as pd

# ---------- 1. Build a reproducible sample dataset ----------
rng = np.random.default_rng(42)
n = 300

catalog = {
    "Laptop":     ("Electronics", 900),
    "Phone":      ("Electronics", 600),
    "Headphones": ("Electronics", 120),
    "Desk":       ("Furniture", 350),
    "Chair":      ("Furniture", 180),
    "Notebook":   ("Stationery", 5),
    "Pen Set":    ("Stationery", 15),
}
products = rng.choice(list(catalog), size=n, p=[.08, .12, .18, .06, .10, .26, .20])

df = pd.DataFrame({
    "order_id": np.arange(1001, 1001 + n),
    "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "region": rng.choice(["North", "South", "East", "West"], size=n),
    "product": products,
    "quantity": rng.integers(1, 6, size=n),
})
df["category"] = df["product"].map(lambda p: catalog[p][0])
df["unit_price"] = df["product"].map(lambda p: catalog[p][1])
df["sales"] = df["quantity"] * df["unit_price"]
df["month"] = df["date"].dt.to_period("M")

# ---------- 2. Key metrics ----------
total_sales = df["sales"].sum()
aov = df["sales"].mean()
n_orders = df["order_id"].nunique()
units = df["quantity"].sum()

print("=== KEY METRICS ===")
print(f"Orders:              {n_orders}")
print(f"Units sold:          {units}")
print(f"Total sales:         ${total_sales:,.0f}")
print(f"Average order value: ${aov:,.2f}")

# ---------- 3. Top products ----------
top_products = (df.groupby("product")
                  .agg(revenue=("sales", "sum"), units=("quantity", "sum"), orders=("order_id", "count"))
                  .sort_values("revenue", ascending=False))
top_products["share_%"] = (top_products["revenue"] / total_sales * 100).round(1)
print("\n=== TOP PRODUCTS (by revenue) ===")
print(top_products)

# ---------- 4. Group summaries ----------
by_category = (df.groupby("category")
                 .agg(revenue=("sales", "sum"), orders=("order_id", "count"), aov=("sales", "mean"))
                 .round(2).sort_values("revenue", ascending=False))
print("\n=== BY CATEGORY ===")
print(by_category)

by_region = (df.groupby("region")
               .agg(revenue=("sales", "sum"), orders=("order_id", "count"), aov=("sales", "mean"))
               .round(2).sort_values("revenue", ascending=False))
print("\n=== BY REGION ===")
print(by_region)

monthly = df.groupby("month")["sales"].sum()
print("\n=== MONTHLY SALES ===")
print(monthly)

quarterly = df.groupby(df["date"].dt.to_period("Q"))["sales"].sum()
print("\n=== QUARTERLY SALES ===")
print(quarterly)

pivot = df.pivot_table(index="region", columns="category", values="sales", aggfunc="sum", fill_value=0)
print("\n=== REGION x CATEGORY (revenue) ===")
print(pivot)

# ---------- 5. Highlights ----------
print("\n=== HIGHLIGHTS ===")
print(f"Best month:  {monthly.idxmax()} (${monthly.max():,.0f})")
print(f"Worst month: {monthly.idxmin()} (${monthly.min():,.0f})")
print(f"Top region:  {by_region.index[0]}")
print(f"Top category: {by_category.index[0]} "
      f"({by_category['revenue'].iloc[0] / total_sales:.0%} of sales)")
print(f"Top product: {top_products.index[0]} "
      f"({top_products['share_%'].iloc[0]}% of sales)")

df.to_csv("/mnt/user-data/outputs/sample_sales.csv", index=False)
