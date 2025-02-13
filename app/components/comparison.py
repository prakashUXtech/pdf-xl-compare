import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from app.utils.config import EXPORT_DIR, EXPORT_SETTINGS

class ComparisonComponent:
    def __init__(self):
        self.export_dir = EXPORT_DIR
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_data(self, pdf_df, excel_df):
        """Compare PDF and Excel data."""
        if pdf_df is None or excel_df is None:
            return None
        
        # Ensure both dataframes have the same columns
        required_columns = ['CREDIT_MONTH', 'WAGES']
        for df in [pdf_df, excel_df]:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in dataframe: {missing}")
        
        # Merge dataframes on CREDIT_MONTH
        merged_df = pd.merge(
            pdf_df[required_columns],
            excel_df[required_columns],
            on='CREDIT_MONTH',
            how='outer',
            suffixes=('_pdf', '_excel')
        )
        
        # Calculate differences
        merged_df['WAGES_DIFF'] = merged_df['WAGES_pdf'] - merged_df['WAGES_excel']
        merged_df['MISMATCH'] = abs(merged_df['WAGES_DIFF']) > 0.01  # Allow small floating point differences
        
        # Sort by date
        merged_df['sort_date'] = pd.to_datetime(merged_df['CREDIT_MONTH'].apply(lambda x: f"01/{x}"), format='%d/%m/%Y')
        merged_df = merged_df.sort_values('sort_date')
        merged_df = merged_df.drop('sort_date', axis=1)
        
        return merged_df
    
    def export_results(self, df, format='csv'):
        """Export comparison results to file."""
        if df is None or df.empty:
            raise ValueError("No data to export")
        
        timestamp = datetime.now().strftime(EXPORT_SETTINGS['timestamp_format'])
        filename = EXPORT_SETTINGS['filename_template'].format(
            timestamp=timestamp,
            format=format
        )
        filepath = self.export_dir / filename
        
        # Format data for export
        export_df = df.copy()
        for col in ['WAGES_pdf', 'WAGES_excel', 'WAGES_DIFF']:
            if col in export_df.columns:
                export_df[col] = export_df[col].round(2)
        
        # Export based on format
        if format == 'csv':
            export_df.to_csv(filepath, index=False)
        elif format == 'xlsx':
            export_df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return filepath
    
    def render_comparison(self, merged_df):
        """Render comparison results in Streamlit."""
        if merged_df is None:
            st.warning("No comparison data available")
            return
        
        # Summary statistics
        total_records = len(merged_df)
        mismatches = merged_df['MISMATCH'].sum()
        match_percentage = ((total_records - mismatches) / total_records * 100) if total_records > 0 else 0
        total_diff = abs(merged_df['WAGES_DIFF']).sum()
        
        # Display summary
        st.subheader("Comparison Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", total_records)
        with col2:
            st.metric("Mismatches", int(mismatches))
        with col3:
            st.metric("Match Percentage", f"{match_percentage:.1f}%")
        with col4:
            st.metric("Total Difference", f"₹{total_diff:,.2f}")
        
        # Display detailed comparison
        st.subheader("Detailed Comparison")
        
        # Filter options
        col1, col2 = st.columns([2, 1])
        with col1:
            show_all = st.checkbox("Show all records", value=False)
            if not show_all:
                comparison_df = merged_df[merged_df['MISMATCH']]
            else:
                comparison_df = merged_df
        
        with col2:
            # Export options
            export_format = st.selectbox(
                "Export Format",
                EXPORT_SETTINGS['available_formats'],
                index=0
            )
            
            if st.button("Export Results"):
                try:
                    filepath = self.export_results(comparison_df, format=export_format)
                    st.success(f"Results exported to: {filepath.name}")
                    
                    # Provide download link
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="Download Results",
                            data=f,
                            file_name=filepath.name,
                            mime='application/octet-stream'
                        )
                except Exception as e:
                    st.error(f"Error exporting results: {str(e)}")
        
        if not comparison_df.empty:
            # Style the dataframe
            def highlight_mismatch(row):
                if row['MISMATCH']:
                    return ['background-color: #ffcdd2'] * len(row)
                return [''] * len(row)
            
            # Format and display the data
            display_df = comparison_df.copy()
            display_df['WAGES_pdf'] = display_df['WAGES_pdf'].round(2)
            display_df['WAGES_excel'] = display_df['WAGES_excel'].round(2)
            display_df['WAGES_DIFF'] = display_df['WAGES_DIFF'].round(2)
            
            # Add formatting
            st.dataframe(
                display_df.style
                .apply(highlight_mismatch, axis=1)
                .format({
                    'WAGES_pdf': '₹{:,.2f}',
                    'WAGES_excel': '₹{:,.2f}',
                    'WAGES_DIFF': '₹{:,.2f}'
                }),
                height=400
            )
            
            # Add summary statistics
            st.subheader("Statistical Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("PDF Statistics")
                st.write(display_df['WAGES_pdf'].describe().round(2))
            
            with col2:
                st.write("Excel Statistics")
                st.write(display_df['WAGES_excel'].describe().round(2))
            
        else:
            st.info("No mismatches found!" if not show_all else "No data to display") 