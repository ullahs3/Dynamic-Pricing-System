import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_preprocess_data(filepath="sales_data_sample.csv"):
    # Load the dataset
    df = pd.read_csv(filepath, encoding="latin1")

    # Select relevant columns
    columns_to_use = [
        "PRICEEACH",
        "QUANTITYORDERED",
        "ORDERDATE",
        "PRODUCTLINE",
        "COUNTRY",
        "PRODUCTCODE"
    ]
    df = df[columns_to_use].dropna()

    # Convert ORDERDATE to datetime and extract month
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    df = df.dropna(subset=["ORDERDATE"])
    df["MONTH"] = df["ORDERDATE"].dt.month

    # Normalize PRICEEACH
    price_scaler = MinMaxScaler()
    df["NORMALIZED_PRICE"] = price_scaler.fit_transform(df[["PRICEEACH"]])

    # Encode PRODUCTLINE and COUNTRY
    product_line_encoder = LabelEncoder()
    df["ENCODED_PRODUCTLINE"] = product_line_encoder.fit_transform(df["PRODUCTLINE"])

    country_encoder = LabelEncoder()
    df["ENCODED_COUNTRY"] = country_encoder.fit_transform(df["COUNTRY"])

    # Simulate a current inventory column for gym env
    df["CURRENT_INVENTORY"] = 100  # example placeholder

    # Final features for Gym env
    final_df = df[[
        "NORMALIZED_PRICE",
        "CURRENT_INVENTORY",
        "MONTH",
        "ENCODED_PRODUCTLINE",
        "ENCODED_COUNTRY"
    ]].reset_index(drop=True)
    return final_df
x = load_preprocess_data(filepath="sales_data_sample.csv")
print(x)