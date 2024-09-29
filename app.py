import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


st.set_page_config(page_title="Vehicle Price Prediction_Arya Wahyu Wijaya", layout="wide")

st.title("PREDIKSI HARGA KENDARAAN DI AUSTRALIA")

# Load Dataset
@st.cache_data  # Use the new caching decorator
def load_data():
    # Load the Australian Vehicle Prices dataset
    data = pd.read_csv('Australian Vehicle Prices.csv')
    
    # Clean up data (preprocessing)
    data['Kilometres'] = pd.to_numeric(data['Kilometres'].str.replace(',', '').str.extract('(\d+)', expand=False), errors='coerce')
    data['FuelConsumption'] = pd.to_numeric(data['FuelConsumption'].str.extract('(\d+\.?\d*)', expand=False), errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'].str.replace(',', ''), errors='coerce')
    
    # Drop rows with missing prices
    data_clean = data.dropna(subset=['Price'])
    return data_clean

df = load_data()

# Sidebar for user input
st.sidebar.header('Input Vehicle Features')

# User input for Year
year = st.sidebar.slider('Year', int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].median()))

# User input for Kilometres as a range
km_range = st.sidebar.slider('Kilometres', int(df['Kilometres'].min()), int(df['Kilometres'].max()), (int(df['Kilometres'].min()), int(df['Kilometres'].max())))

# User input for Fuel Consumption as a range
fuel_consumption_range = st.sidebar.slider('Fuel Consumption', float(df['FuelConsumption'].min()), float(df['FuelConsumption'].max()), (float(df['FuelConsumption'].min()), float(df['FuelConsumption'].max())))

# User input for categorical variables
brand = st.sidebar.selectbox('Brand', df['Brand'].unique())
transmission = st.sidebar.selectbox('Transmission', df['Transmission'].unique())
fuel_type = st.sidebar.selectbox('Fuel Type', df['FuelType'].unique())

# Confirmation Button
if st.sidebar.button('Confirm'):
    # Filter the data based on the selected ranges
    filtered_data = df[(df['Year'] == year) & 
                       (df['Kilometres'] >= km_range[0]) & (df['Kilometres'] <= km_range[1]) & 
                       (df['FuelConsumption'] >= fuel_consumption_range[0]) & (df['FuelConsumption'] <= fuel_consumption_range[1]) & 
                       (df['Brand'] == brand) & 
                       (df['Transmission'] == transmission) & 
                       (df['FuelType'] == fuel_type)]
    
    # Check if there are any filtered results
    if not filtered_data.empty:
        # Display the minimum and maximum price above the filtered vehicles
        min_price = filtered_data['Price'].min()
        max_price = filtered_data['Price'].max()
        st.subheader("Predicted Price :")
        st.write(f"Lowest Price: ${min_price:,.2f}")
        st.write(f"Highest Price: ${max_price:,.2f}")

        # Display filtered vehicles
        st.subheader("Filtered Vehicles:")
        st.write(filtered_data)
    else:
        st.subheader("No vehicles found with the selected criteria.")
else:
    # Display the full dataset sample before confirmation
    st.subheader("Dataset Sample")
    st.write(df)

# Customize background and theme (dark mode styling)
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
    }
    .css-1d391kg { 
        background-color: #2E2E2E; 
    }
    .css-1offfwp { 
        background-color: #2E2E2E; 
    }
    h1, h2, h3, p {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
