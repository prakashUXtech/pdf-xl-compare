"""
Service for comparing data between different sources.
"""
import pandas as pd
import streamlit as st
from typing import Dict, Optional
from pathlib import Path

class ComparisonService:
    """Handles data comparison operations."""
    
    @staticmethod
    @st.cache_data
    def compare_data(pdf_data: pd.DataFrame, excel_data: pd.DataFrame) -> pd.DataFrame:
        """Compare PDF extracted data with Excel data."""
        try:
            # Prepare PDF data
            pdf_columns = ['CREDIT_MONTH']
            if 'WAGES' in pdf_data.columns:
                pdf_columns.append('WAGES')
            if 'PEN.WAGES' in pdf_data.columns:
                pdf_columns.append('PEN.WAGES')
            
            pdf_compare = pdf_data[pdf_columns].copy()
            
            # Standardize PDF column names
            pdf_compare = pdf_compare.rename(columns={
                'WAGES': 'WAGES_PDF',
                'PEN.WAGES': 'PEN.WAGES_PDF'
            })
            
            # Find matching columns in Excel data
            month_cols = [col for col in excel_data.columns if 'MONTH' in col.upper()]
            wages_cols = [col for col in excel_data.columns if 'WAGE' in col.upper() and 'PEN' not in col.upper()]
            
            if not month_cols or not wages_cols:
                st.error(f"""
                Missing required columns in Excel file:
                - Month column (found: {month_cols})
                - Wages column (found: {wages_cols})
                """)
                return pd.DataFrame()
            
            # Prepare Excel data
            excel_compare = excel_data[[month_cols[0], wages_cols[0]]].copy()
            excel_compare = excel_compare.rename(columns={
                month_cols[0]: 'CREDIT_MONTH',
                wages_cols[0]: 'WAGES_EXCEL'
            })
            
            # Clean and standardize data
            for df in [pdf_compare, excel_compare]:
                # Convert wages to numeric
                wage_cols = [col for col in df.columns if 'WAGE' in col.upper()]
                for col in wage_cols:
                    df[col] = pd.to_numeric(df[col].astype(str).replace('[^\d.-]', '', regex=True), errors='coerce')
                
                # Standardize date format
                df['CREDIT_MONTH'] = pd.to_datetime(df['CREDIT_MONTH'].astype(str).str.strip(), format='%m/%Y', errors='coerce')
                df['CREDIT_MONTH'] = df['CREDIT_MONTH'].dt.strftime('%m/%Y')
            
            # Merge dataframes
            comparison = pd.merge(
                pdf_compare,
                excel_compare,
                on='CREDIT_MONTH',
                how='outer'
            )
            
            # Calculate differences
            if 'WAGES_PDF' in comparison.columns and 'WAGES_EXCEL' in comparison.columns:
                comparison['DIFFERENCE'] = comparison['WAGES_PDF'] - comparison['WAGES_EXCEL']
                
                # Mark mismatches (using a small threshold to account for rounding)
                threshold = 0.01  # 1 paisa threshold
                comparison['MISMATCH'] = abs(comparison['DIFFERENCE']) > threshold
            
            # Sort by date
            comparison['sort_date'] = pd.to_datetime(comparison['CREDIT_MONTH'], format='%m/%Y')
            comparison = comparison.sort_values('sort_date')
            comparison = comparison.drop('sort_date', axis=1)
            
            return comparison
            
        except Exception as e:
            st.error(f"Error comparing data: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return pd.DataFrame()
    
    @staticmethod
    def show_comparison_results(comparison: pd.DataFrame, excel_filename: str = None) -> None:
        """Display comparison results in the UI."""
        if comparison.empty:
            return
        
        # Show results
        st.subheader("Comparison Results")
        
        # Calculate mismatches
        mismatches = comparison[comparison['MISMATCH']]
        if not mismatches.empty:
            st.error(f"Found {len(mismatches)} mismatches!")
            
            # Show summary statistics for mismatches
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Mismatches", len(mismatches))
            
            # Show wage differences if they exist
            if 'DIFFERENCE' in mismatches.columns:
                with cols[1]:
                    st.metric("Total Wages Difference", 
                             f"₹{abs(mismatches['DIFFERENCE']).sum():,.2f}",
                             help="Difference in regular wages")
            
            # Show combined differences
            with cols[3]:
                total_diff = abs(mismatches['DIFFERENCE']).sum()
                st.metric("Total Combined Difference", f"₹{total_diff:,.2f}")
            
            # Show mismatches first
            st.subheader("Mismatched Records")
            
            # Format the numeric columns for display
            display_mismatches = mismatches.copy()
            numeric_cols = [col for col in display_mismatches.columns 
                          if any(x in col for x in ['WAGES', 'DIFFERENCE'])]
            
            for col in numeric_cols:
                display_mismatches[col] = display_mismatches[col].apply(
                    lambda x: f"₹{x:,.2f}" if pd.notnull(x) else ''
                )
            
            st.dataframe(
                display_mismatches.drop(columns=['MISMATCH']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No mismatches found! All wages match perfectly.")
        
        # Show all records in an expander
        with st.expander("View All Records"):
            # Format the numeric columns for display
            display_df = comparison.copy()
            numeric_cols = [col for col in display_df.columns 
                          if any(x in col for x in ['WAGES', 'DIFFERENCE'])]
            
            for col in numeric_cols:
                display_df[col] = display_df[col].apply(
                    lambda x: f"₹{x:,.2f}" if pd.notnull(x) else ''
                )
            
            st.dataframe(
                display_df.drop(columns=['MISMATCH']),
                use_container_width=True,
                hide_index=True
            )
        
        # Generate download filename based on Excel filename
        if excel_filename:
            base_name = Path(excel_filename).stem
            download_filename = f"{base_name}_comparison.csv"
        else:
            download_filename = "comparison_results.csv"
        
        # Download comparison
        st.download_button(
            "Download Full Comparison",
            comparison.drop(columns=['MISMATCH']).to_csv(index=False),
            download_filename,
            "text/csv",
            help="Download the complete comparison including matches and mismatches"
        ) 