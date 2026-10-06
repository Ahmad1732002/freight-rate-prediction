import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df=pd.read_csv("train-test.csv")
#weight column has 300 null values, and 292 negative values
#market index column has 374 null values
df_clean=df.dropna(subset=["weight"]).copy()
df_clean=df_clean[df_clean["weight"]>0]
df_clean=df_clean.dropna(subset=["market_index"])

#drop id column as it will not help in training
df_clean=df_clean.drop(columns=["load_id"])

#convert date objects to datetime objects
df_clean["date"]=pd.to_datetime(df_clean["date"])
df_clean=df_clean.sort_values(by="date")

# monthly_counts = (
#     df_clean["date"]
#     .dt.to_period("M")
#     .value_counts()
#     .sort_index()
# )

# print(monthly_counts)

# monthly_counts.plot(kind="line", marker="o")

# plt.xlabel("Month")
# plt.ylabel("Number of loads")
# plt.title("Number of loads per month")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.savefig("monthly_loads.png")
#use chronological split to create train and test sets, instead of random split to avoid data leakage
train = df_clean[df_clean["date"] < "2025-09-01"].copy()
test = df_clean[df_clean["date"] >= "2025-09-01"].copy()

X_train=train.drop(columns=["posted_rate"])
X_test=test.drop(columns=["posted_rate"])

y_train=train["posted_rate"]
y_test=test["posted_rate"]



print(train.shape)
print(test.shape)

##Processing data features for training
#We dont need all info in the date column we can extract month and day of week from it, and drop the date column
for df in [X_train, X_test]:
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

X_train = X_train.drop(columns=["date"])
X_test = X_test.drop(columns=["date"])
# breakpoint()
#add route as categorical feature, as it is a combination of pickup and delivery
# X_train["route"] = X_train["pickup"] + "_" + X_train["delivery"]
# X_test["route"] = X_test["pickup"] + "_" + X_test["delivery"]

# X_train = X_train.drop(columns=["pickup", "delivery"])
# X_test = X_test.drop(columns=["pickup", "delivery"])

cat_features = [
    "pickup",
    "delivery",
    "equipment"
]



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
    cat_features=cat_features,
)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = (
    np.abs((y_test - pred) / y_test).mean() * 100
)

print("MAPE:", mape)

print("MAE:", mae)

print("RMSE:", rmse)

valid_df=pd.read_csv("validation.csv")

X_validation = valid_df.copy()


X_validation["date"] = pd.to_datetime(X_validation["date"])

X_validation["month"] = X_validation["date"].dt.month
X_validation["day_of_week"] = X_validation["date"].dt.dayofweek

X_validation = X_validation.drop(columns=["date","load_id"])

pred = model.predict(X_validation)
valid_df['predicted_rate']=pred


selected = valid_df[["load_id", "predicted_rate"]]
selected.to_csv("validation_predictions.csv", index=False)

#December predictions are made in dec_eval.py, since the December file
#only has pickup, delivery, distance, equipment, weight and date



