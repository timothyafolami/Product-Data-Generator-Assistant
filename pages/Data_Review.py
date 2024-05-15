import streamlit as st
import pandas as pd
import time
import os

# Welcome the user to the page
st.title("Data Preview and Download")
st.header("Congratulations! Your data has been generated successfully.")
st.markdown("Preview and download your data below.")

# Check for the presence of data_path
data_path = './data.csv'
exist = os.path.exists(data_path)

# Add a loading status indicator
if not exist:
    st.warning("Please generate your data by interacting with the chat app.")
else:
    st.success("Your data is ready to preview and download!")

# Display the data using Streamlit DataFrame
if exist:
    st.header("Your Data")
    st.write("Congratulations, your data has been successfully generated!")
    df = pd.read_csv(data_path)
    st.dataframe(df)

    # Add a download button
    st.download_button("Download Data", data_path, "data.csv", "text/csv", key='download-csv')
    
else:
    st.warning("Error: No data found. Please go back to the chat app to generate your data.")

# Add a progress bar
progress = st.progress(0)
for i in range(100):
    progress.progress(i+1)
    time.sleep(0.01)  # adjust the speed of the progress bar

if exist:
    st.balloons()