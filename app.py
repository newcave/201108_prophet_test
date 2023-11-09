import streamlit as st
import pandas as pd
from prophet import Prophet

# Load the data from the local file and perform the necessary processing
file_path = "co_data2.csv"

try:
    data = pd.read_csv(file_path)
    
    # Check the columns available in the DataFrame
    st.write(data.columns)
    
    # Assuming 'Date' and 'Confirmed' are the correct column names in the CSV file
    # Modify the code to use the correct column names for filtering
    df_korea = data[data['CorrectColumnNameForCountry'] == 'Korea, South']  # Replace 'CorrectColumnNameForCountry' with the actual column name
    
    # Other parts of your code...
    # ...
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except pd.errors.ParserError as e:
    st.error(f"Error occurred while reading CSV: {e}")

# Continue with the rest of the code

if 'data' in locals():
    # Check the columns in the data to ensure correct column names
    st.write(data.columns)

    # Assuming that the columns may be named 'Date' and 'Confirmed'
    df_korea = data[data['Country/Region'] == 'Korea, South']  # Replace 'Country/Region' with correct column name
    df_korea = df_korea[['Date', 'Confirmed']]  # Update column names based on the actual column names
    df_korea.columns = ['ds', 'y']

    # Create a Streamlit app
    st.title('COVID-19 Confirmed Cases Prediction for South Korea')

    # Display the data
    st.subheader('Data')
    st.write(df_korea)

    # Create and fit the Prophet model
    m = Prophet()
    m.fit(df_korea)

    # Forecast future dates
    future = m.make_future_dataframe(periods=14)
    forecast = m.predict(future)

# Plot the forecast
st.subheader('Forecast')
st.write(forecast)

st.write("### Visualizing the Forecast")
st.write("Use the slider to change the date range.")

# Select a date range with a slider
date_range = st.slider("Select Date Range", min_value=df_korea['ds'].min(), max_value=df_korea['ds'].max())

# Plotting the forecast
st.write(f"""
    ### Forecast Plot (Up to {date_range})
""")
fig = m.plot(forecast)
st.pyplot(fig)

# Show changepoints
st.subheader('Changepoints')
st.write("Changepoints detected by the model")
fig_changepoints = m.plot(forecast)
a = Prophet._plot_changepoints(fig_changepoints.gca(), m, forecast)
st.pyplot(fig_changepoints)
