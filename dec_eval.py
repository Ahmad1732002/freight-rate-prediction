import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# 1. Load training data
# =========================

df = pd.read_csv("train-test.csv")

# Clean data
df_clean = df.dropna(subset=["weight"]).copy()
df_clean = df_clean[df_clean["weight"] > 0]
df_clean = df_clean.dropna(subset=["market_index"])

# We don't need load_id
df_clean = df_clean.drop(columns=["load_id"])

# Convert date
df_clean["date"] = pd.to_datetime(df_clean["date"])

# Sort chronologically
df_clean = df_clean.sort_values("date")


# =========================
# 2. Train/test split
# =========================

train = df_clean[df_clean["date"] < "2025-09-01"].copy()
test = df_clean[df_clean["date"] >= "2025-09-01"].copy()


# =========================
# 3. Use ONLY features
#    available in December
# =========================

features = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date"
]

X_train = train[features].copy()
X_test = test[features].copy()

y_train = train["posted_rate"]
y_test = test["posted_rate"]


# =========================
# 4. Create date features
# =========================

for data in [X_train, X_test]:
    data["month"] = data["date"].dt.month
    # day_of_month removed: it learned noise and caused a fake drop mid-December
    data["day_of_week"] = data["date"].dt.dayofweek

X_train = X_train.drop(columns=["date"])
X_test = X_test.drop(columns=["date"])


# =========================
# 5. CatBoost categorical features
# =========================

cat_features = [
    "pickup",
    "delivery",
    "equipment"
]


# =========================
# 6. Train model
# =========================

model = CatBoostRegressor(
    iterations=250,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=100,
    random_seed=42
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)


# =========================
# 7. Evaluate
# =========================

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

mape = (
    np.abs((y_test - pred) / y_test).mean() * 100
)

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)


# =========================
# 8. Predict December
# =========================

dec_df = pd.read_csv("december-chart-inputs.csv")

december_df = dec_df[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date"
    ]
].copy()

december_df["date"] = pd.to_datetime(december_df["date"])

# Create same features as training
december_df["month"] = december_df["date"].dt.month
december_df["day_of_week"] = december_df["date"].dt.dayofweek

# Remove date because model doesn't use raw date
december_df = december_df.drop(columns=["date"])


# =========================
# 9. Predict
# =========================

december_pred = model.predict(december_df)

dec_df["predicted_rate"] = december_pred


# =========================
# 10. Save
# =========================

dec_df.to_csv(
    "data/december_chart_inputs.csv",
    index=False
)

print("December predictions saved!")