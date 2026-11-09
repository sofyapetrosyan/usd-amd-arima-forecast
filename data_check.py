import pandas as pd

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

print("Տողերի քանակը՝", len(data))
print("\nԱռաջին 5 տողը՝")
print(data.head())
print("\nՎերջին 5 տողը՝")
print(data.tail())
