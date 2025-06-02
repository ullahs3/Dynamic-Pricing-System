import streamlit as st
import pandas as pd
import plotly.express as px

# Load your dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

# Map numeric MONTH_ID to actual month names
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

df = df.dropna(subset=["COUNTRY", "PRODUCTLINE", "SALES", "MONTH_ID"])
df["MONTH"] = df["MONTH_ID"].map(month_map)
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df["MONTH"] = pd.Categorical(df["MONTH"], categories=month_order, ordered=True)


st.title("📊 Sales Dashboard")

tab1, tab2, tab3 = st.tabs(["🌍 Country Sales", "👥 Top Customers", "🚗 Product Category Breakdown"])

# 🌍 Tab 1: Country Sales
with tab1:
    st.subheader("Countries Ranked by Sales")
    country_sales = df.groupby("COUNTRY")["SALES"].sum().reset_index().sort_values(by="SALES", ascending=False)
    fig_country = px.bar(
        country_sales,
        x="SALES",
        y="COUNTRY",
        orientation='h',
        title="Countries Ranked by Sales",
        color="SALES",
        color_continuous_scale="Tealrose"
    )
    st.plotly_chart(fig_country)

# 👥 Tab 2: Top Customers
with tab2:
    st.subheader("Top 5 Quarterly Active Customers / Month")

    # Recalculate QUARTER and MONTH to ensure they exist
    df["QUARTER"] = pd.cut(df["MONTH_ID"], bins=[0,3,6,9,12], labels=["Q1", "Q2", "Q3", "Q4"])
    df["MONTH"] = df["MONTH_ID"].map(month_map)
    df["MONTH"] = pd.Categorical(df["MONTH"], categories=month_order, ordered=True)

    # Get top 5 per month by sales
    top_customers = (
        df.groupby(["QUARTER", "MONTH", "CUSTOMERNAME"])["SALES"]
        .sum()
        .reset_index()
        .sort_values(["QUARTER", "MONTH", "SALES"], ascending=[True, True, False])
    )

    top_5_customers = top_customers.groupby(["QUARTER", "MONTH"]).head(5)

    # Format
    top_5_customers["SALES"] = top_5_customers["SALES"].apply(lambda x: f"${x:,.0f}")
    top_5_customers = top_5_customers.rename(columns={
        "CUSTOMERNAME": "Customer",
        "MONTH": "Month",
        "QUARTER": "Quarter",
        "SALES": "Total Sales"
    })

    # Display as a sorted table
    st.dataframe(
        top_5_customers[["Quarter", "Month", "Customer", "Total Sales"]],
        use_container_width=True
    )


# 🚗 Tab 3: Sales Category Breakdown
with tab3:
    st.subheader("Monthly Order Quantity by Product Line")
    df["TimeLabel"] = df["YEAR_ID"].astype(str) + " - Q" + df["QUARTER"].astype(str) + " - " + df["MONTH"].astype(str)
    product_filter = st.selectbox("Select a Product Category", options=["All"] + sorted(df["PRODUCTLINE"].unique()))
    filtered_df = df if product_filter == "All" else df[df["PRODUCTLINE"] == product_filter]

    category_data = filtered_df.groupby(["TimeLabel", "PRODUCTLINE"])["QUANTITYORDERED"].sum().reset_index()
    category_data["TimeLabel"] = pd.Categorical(category_data["TimeLabel"],
        categories=sorted(df["TimeLabel"].unique(), key=lambda x: (
            x.split(" - ")[0], x.split(" - ")[1], month_order.index(x.split(" - ")[2])
        )), ordered=True)

    fig = px.bar(
        category_data,
        x="QUANTITYORDERED",
        y="TimeLabel",
        color="PRODUCTLINE",
        orientation="h",
        title="Sales Category by Month and Quarter",
        labels={"QUANTITYORDERED": "Order Quantity", "TimeLabel": "Year - Quarter - Month"},
        height=1000
    )
    fig.update_layout(barmode="stack", yaxis={'categoryorder':'array', 'categoryarray': category_data["TimeLabel"].cat.categories})
    st.plotly_chart(fig, use_container_width=True)
