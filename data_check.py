import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

split_index = int(len(data) * 0.8)
train = data.iloc[:split_index]
test = data.iloc[split_index:].copy()

last_train_value = train["usd_amd"].iloc[-1]
test["naive_forecast"] = last_train_value
errors = test["usd_amd"] - test["naive_forecast"]
mae = errors.abs().mean()
rmse = (errors.pow(2).mean()) ** 0.5

print("Պարզ կանխատեսման MAE՝", round(mae, 3))
print("Պարզ կանխատեսման RMSE՝", round(rmse, 3))
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
plt.axvline(test["date"].min(), color="gray", linestyle=":", label="Train/test split")
plt.title("USD/AMD: Training, Test and Baseline Forecast")
plt.xlabel("Date")
plt.ylabel("AMD per 1 USD")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(project_dir / "usd_amd_exchange_rate.png", dpi=160)
print("Գրաֆիկը պահպանվել է՝ usd_amd_exchange_rate.png")

data["daily_change"] = data["usd_amd"].diff()

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

