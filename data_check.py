import pandas as pd
import matplotlib.pyplot as plt

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

print("Տողերի քանակը՝", len(data))
print("Առաջին օրը՝", data["date"].min().date())
print("Վերջին օրը՝", data["date"].max().date())

plt.figure(figsize=(11, 5))
plt.plot(data["date"], data["usd_amd"], color="#2563eb", linewidth=1.5)
plt.title("USD/AMD Exchange Rate (2023–2025)")
plt.xlabel("Date")
plt.ylabel("AMD per 1 USD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("usd_amd_exchange_rate.png", dpi=160)
print("Գրաֆիկը պահպանվել է՝ usd_amd_exchange_rate.png")
