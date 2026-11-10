import numpy as np
import pandas as pd
from scipy import stats

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
           "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", 
           "length", "width", "height", "curb-weight", "engine-type", 
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", 
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df = pd.read_csv(url, header=None, names=headers)

df.replace("?", np.nan, inplace=True)

numeric_cols = ["bore", "stroke", "horsepower", "price", "wheel-base"]
for col in numeric_cols:
  df[col] = pd.to_numeric(df[col])

print("Кореляційна матриця")
cols_1 = ["bore", "stroke", "compression-ratio", "horsepower"]
print(df[cols_1].corr())

stroke_price_corr = df[["stroke", "price"]].corr().iloc[0, 1]
print(
    f"\nКореляція між 'stroke' та 'price'\nr = {stroke_price_corr:.4f}"
)

print("\nСередня ціна за типом кузова")
avg_price_body = df.groupby("body-style")["price"].mean()
print(avg_price_body)

clean_wb = df[["wheel-base", "price"]].dropna()
pearson_coef_wb, p_value_wb = stats.pearsonr(
    clean_wb["wheel-base"], clean_wb["price"]
)
print("\nPearson & P-value: 'wheel-base' vs 'price'")
print(f"Pearson correlation: {pearson_coef_wb:.4f}")
print(f"P-value: {p_value_wb:.4e}")

clean_hp = df[["horsepower", "price"]].dropna()
pearson_coef_hp, p_value_hp = stats.pearsonr(
    clean_hp["horsepower"], clean_hp["price"]
)
print("\nPearson & P-value: 'horsepower' vs 'price'")
print(f"Pearson correlation: {pearson_coef_hp:.4f}")
print(f"P-value: {p_value_hp:.4e}")

print("\nДисперсійний аналіз (ANOVA)")

clean_drive = df[["drive-wheels", "price"]].dropna()
grouped_drive = [
    group["price"].values
    for name, group in clean_drive.groupby("drive-wheels")
]
f_val_drive, p_val_drive = stats.f_oneway(*grouped_drive)
print(
    f"ANOVA 'drive-wheels' vs 'price': F = {f_val_drive:.4f}, P-value ="
    f" {p_val_drive:.4e}"
)

clean_body = df[["body-style", "price"]].dropna()
grouped_body = [
    group["price"].values for name, group in clean_body.groupby("body-style")
]
f_val_body, p_val_body = stats.f_oneway(*grouped_body)
print(
    f"ANOVA 'body-style' vs 'price':   F = {f_val_body:.4f}, P-value ="
    f" {p_val_body:.4e}"
)