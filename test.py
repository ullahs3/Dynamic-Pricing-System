import pandas as pd

df_raw = pd.read_csv("sales_data_sample.csv", encoding="latin1")
print(df_raw["COUNTRY"].unique())