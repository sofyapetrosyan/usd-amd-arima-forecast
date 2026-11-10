import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

project_dir = Path(__file__).resolve().parent
file_path = r"C:\Users\SOFI\Desktop\USD_AMD_Forecast\usd_amd_data.xlsx"

data = pd.read_excel(
    file_path,
    sheet_name="USD_AMD_Dataset ",
    header=None,
    skiprows=1,
    usecols=[0, 1],
    names=["date", "usd_amd"],
)

data["date"] = pd.to_datetime(data["date"])
data = data.dropna(subset=["date", "usd_amd"]).sort_values("date")

# Keep the final 20% for the main evaluation; tune model orders inside the first 80%.
split_index = int(len(data) * 0.8)
train = data.iloc[:split_index]
test = data.iloc[split_index:].copy()
train_values = train["usd_amd"].astype(float)

# ADF test on training data only.
adf_original = adfuller(train_values, maxlag=5, autolag=None, result_object=False)
adf_difference = adfuller(
    train_values.diff().dropna(), maxlag=5, autolag=None, result_object=False
)
print("Սկզբնական շարքի ADF p-value՝", round(adf_original[1], 4))
print("Առաջին տարբերության ADF p-value՝", round(adf_difference[1], 4))

# Use the last 20% of training data as a validation period for choosing p and q.
validation_split = int(len(train_values) * 0.8)
validation_train = train_values.iloc[:validation_split]
validation_values = train_values.iloc[validation_split:]
candidate_orders = [
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (3, 1, 0),
    (4, 1, 0),
    (0, 1, 2),
    (0, 1, 3),
    (1, 1, 3),
    (2, 1, 2),
]
validation_results = []

for order in candidate_orders:
    validation_fit = ARIMA(validation_train, order=order).fit()
    validation_forecast = validation_fit.forecast(steps=len(validation_values)).to_numpy()
    validation_errors = validation_values.to_numpy() - validation_forecast
    validation_mae = abs(validation_errors).mean()
    validation_rmse = (validation_errors ** 2).mean() ** 0.5
    validation_results.append((validation_mae, order))
    print(
        "ARIMA", order,
        "validation MAE՝", round(validation_mae, 3),
        "validation RMSE՝", round(validation_rmse, 3),
    )

best_order = min(validation_results, key=lambda result: result[0])[1]
print("\nValidation-ով ընտրված կարգը՝", best_order)


final_fit = ARIMA(train_values, order=best_order).fit()
residuals = final_fit.resid[1:]

ljung_box = acorr_ljungbox(residuals, lags=[10], model_df=best_order[0] + best_order[2], return_df=True)
print("Մնացորդների Ljung-Box p-value՝", round(ljung_box["lb_pvalue"].iloc[0], 4))
test["arima_forecast"] = final_fit.forecast(steps=len(test)).to_numpy()
arima_errors = test["usd_amd"].to_numpy() - test["arima_forecast"].to_numpy()
arima_mae = abs(arima_errors).mean()
arima_rmse = (arima_errors ** 2).mean() ** 0.5

last_train_value = train_values.iloc[-1]
test["naive_forecast"] = last_train_value
naive_errors = test["usd_amd"].to_numpy() - test["naive_forecast"].to_numpy()
naive_mae = abs(naive_errors).mean()
naive_rmse = (naive_errors ** 2).mean() ** 0.5

print("\nՎերջնական ARIMA", best_order, "MAE՝", round(arima_mae, 3))
print("Վերջնական ARIMA", best_order, "RMSE՝", round(arima_rmse, 3))
print("Պարզ կանխատեսման MAE՝", round(naive_mae, 3))
print("Պարզ կանխատեսման RMSE՝", round(naive_rmse, 3))
print("Ուսուցման տվյալների քանակը՝", len(train))
print("Ստուգման տվյալների քանակը՝", len(test))
print("Ստուգման շրջանի սկիզբը՝", test["date"].min().date())
print("Տողերի քանակը՝", len(data))
print("Առաջին օրը՝", data["date"].min().date())
print("Վերջին օրը՝", data["date"].max().date())

plt.figure(figsize=(11, 5))
plt.plot(train["date"], train["usd_amd"], label="Training data", color="#2563eb")
plt.plot(test["date"], test["usd_amd"], label="Actual test data", color="#111827")
plt.plot(test["date"], test["naive_forecast"], label="Naive forecast", color="red", linestyle="--")
plt.plot(
    test["date"],
    test["arima_forecast"],
    label=f"ARIMA{best_order} forecast",
    color="green",
)
plt.axvline(test["date"].min(), color="gray", linestyle=":", label="Train/test split")
plt.title("USD/AMD: Training, Test and Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("AMD per 1 USD")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(project_dir / "usd_amd_exchange_rate.png", dpi=160)
print("Գրաֆիկը պահպանվել է՝ usd_amd_exchange_rate.png")

data["daily_change"] = data["usd_amd"].diff()
data["abs_change"] = data["daily_change"].abs()

largest_changes = data.nlargest(10, "abs_change")[
    ["date", "usd_amd", "daily_change"]
]

print("\nԱմենամեծ օրական փոփոխությունները՝")
print(largest_changes.to_string(index=False))
plt.figure(figsize=(11, 5))
plt.plot(data["date"], data["daily_change"], color="purple", linewidth=1)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Daily Change in USD/AMD")
plt.xlabel("Date")
plt.ylabel("Change in AMD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(project_dir / "usd_amd_daily_changes.png", dpi=160)
print("Օրական փոփոխությունների գրաֆիկը պահպանվել է՝ usd_amd_daily_changes.png")

plt.figure(figsize=(11, 4))
plt.plot(train["date"].iloc[1:], residuals, color="teal", linewidth=1)
plt.axhline(0, color="black", linestyle="--")
plt.title(f"Residuals of ARIMA{best_order}")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(project_dir / "arima_residuals.png", dpi=160)

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(residuals, lags=30, zero=False, ax=ax)
ax.set_title(f"Residual ACF for ARIMA{best_order}")
plt.tight_layout()
plt.savefig(project_dir / "arima_residual_acf.png", dpi=160)