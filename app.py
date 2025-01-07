import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set title for the web app
st.title("Analisis Data COVID-19 Berbasis Web")

# Upload file
uploaded_file = "data/us-counties.xlsx"

if uploaded_file:
    try:
        # Read the Excel file
        data = pd.read_excel(uploaded_file)

        # Display the first few rows of the data
        st.subheader("Preview Data")
        st.write(data.head())

        # Check if required columns exist
        required_columns = {'state', 'date', 'cases', 'deaths'}
        if not required_columns.issubset(data.columns):
            st.error(f"Dataset harus memiliki kolom: {required_columns}")
        else:
            # Ensure 'date' column is datetime
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            if data['date'].isna().any():
                st.warning("Beberapa data di kolom 'date' tidak valid dan akan dihapus.")
                data = data.dropna(subset=['date'])

            # Sidebar for user input
            state = st.sidebar.selectbox("Pilih Negara Bagian", data['state'].unique())

            # Filter data for the selected state
            state_data = data[data['state'] == state].groupby('date')[['cases', 'deaths']].sum().reset_index()

            # Plot time trends
            st.subheader(f"Tren Kasus COVID-19 di {state}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(state_data['date'], state_data['cases'], label='Kasus', color='blue')
            ax.plot(state_data['date'], state_data['deaths'], label='Kematian', color='red')
            ax.set_title(f"Tren Kasus dan Kematian di {state}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah")
            ax.legend()
            st.pyplot(fig)

            # Calculate national statistics
            total_cases = data['cases'].sum()
            total_deaths = data['deaths'].sum()
            st.subheader("Statistik Nasional")
            st.write(f"Total Kasus di AS: {total_cases}")
            st.write(f"Total Kematian di AS: {total_deaths}")

            # Display top 10 states by cases
            summary = data.groupby('state')[['cases', 'deaths']].sum().reset_index()
            summary = summary.sort_values(by='cases', ascending=False)
            st.subheader("10 Negara Bagian dengan Kasus Terbanyak")
            st.write(summary.head(10))

            # Daily growth analysis
            if 'county' in data.columns:
                data['daily_cases'] = data.groupby('county')['cases'].diff().fillna(0)
                national_daily_cases = data.groupby('date')['daily_cases'].sum()
                peak_cases = national_daily_cases.max()
                peak_date = national_daily_cases.idxmax()
                st.subheader("Puncak Kasus Harian")
                st.write(f"Puncak kasus harian: {int(peak_cases)} kasus pada {peak_date}")

            # Regression analysis
            st.subheader("Prediksi Kasus dengan Regresi Linier")
            data['date_numeric'] = data['date'].map(pd.Timestamp.toordinal)
            X = data[['date_numeric']]
            y = data['cases']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error (MSE): {mse}")

            # Plot predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_test, y_test, color='blue', label='Data Aktual')
            ax.scatter(X_test, y_pred, color='red', label='Prediksi', alpha=0.5)
            ax.set_title("Prediksi Kasus COVID-19")
            ax.set_xlabel("Tanggal (numeric)")
            ax.set_ylabel("Jumlah Kasus")
            ax.legend()
            st.pyplot(fig)

            # Geospatial analysis
            st.subheader("Distribusi Kasus COVID-19 di AS")
            url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
            gdf = gpd.read_file(url)
            state_cases = data.groupby('state')[['cases']].sum().reset_index()
            gdf['name'] = gdf['name'].str.strip()
            merged = gdf.merge(state_cases, left_on='name', right_on='state', how='left')

            # Plot geospatial data
            fig, ax = plt.subplots(figsize=(15, 10))
            merged.plot(column='cases', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
            ax.set_title("Distribusi Kasus COVID-19 di Amerika Serikat")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Footer
st.write("Aplikasi ini dibuat menggunakan Streamlit.")
