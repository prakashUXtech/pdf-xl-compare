import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px

st.set_page_config(layout="wide")
st.title('Contribution Details Viewer')

# Get list of all CSV files in current directory
csv_files = glob.glob('ocr_results_*.csv')

# Sort files by modification time (newest first)
csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

if not csv_files:
    st.warning('No OCR result files found!')
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a dropdown to select the file
        selected_file = st.selectbox('Select OCR Result File:', csv_files)
        
        # Read and display the CSV
        df = pd.read_csv(selected_file)
        st.dataframe(df, use_container_width=True)
        
        # Add download button
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name=selected_file,
            mime='text/csv'
        )
    
    with col2:
        # Add some visualizations
        st.subheader("Data Visualization")
        
        try:
            # Monthly Contribution Chart
            fig1 = px.line(df, x='CREDIT_MONTH', y=['EE', 'ER', 'Pension'],
                          title='Monthly Contribution Breakdown')
            st.plotly_chart(fig1)
            
            # Status Distribution
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts()
                fig2 = px.pie(values=status_counts.values, names=status_counts.index,
                            title='Status Distribution')
                st.plotly_chart(fig2)
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
            st.write("Please ensure the CSV file has the correct column structure") 