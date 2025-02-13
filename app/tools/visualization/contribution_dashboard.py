"""
Streamlit-based dashboard for visualizing contribution data.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContributionDashboard:
    """Dashboard for visualizing contribution data."""
    
    def __init__(self, data_dir: Path = Path('data/processed')):
        """Initialize dashboard with data directory."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def get_available_files(self) -> List[Path]:
        """Get list of available CSV files."""
        return sorted(
            self.data_dir.glob('*.csv'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
    
    def load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def create_contribution_chart(self, df: pd.DataFrame) -> Optional[Union[go.Figure, None]]:
        """Create monthly contribution breakdown chart."""
        try:
            fig = px.line(
                df,
                x='CREDIT_MONTH',
                y=['EE', 'ER', 'Pension'],
                title='Monthly Contribution Breakdown'
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating contribution chart: {e}")
            return None
    
    def create_status_chart(self, df: pd.DataFrame) -> Optional[Union[go.Figure, None]]:
        """Create status distribution chart."""
        try:
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title='Status Distribution'
                )
                return fig
        except Exception as e:
            logger.error(f"Error creating status chart: {e}")
        return None
    
    def run(self):
        """Run the dashboard."""
        st.set_page_config(layout="wide")
        st.title('Contribution Details Dashboard')
        
        # Get available files
        csv_files = self.get_available_files()
        if not csv_files:
            st.warning('No contribution data files found!')
            return
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File selection
            selected_file = st.selectbox(
                'Select Data File:',
                csv_files,
                format_func=lambda x: x.name
            )
            
            # Load and display data
            df = self.load_data(selected_file)
            if df is not None:
                st.dataframe(df, use_container_width=True)
                
                # Download button
                st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False),
                    file_name=selected_file.name,
                    mime='text/csv'
                )
        
        with col2:
            st.subheader("Data Visualization")
            
            if df is not None:
                # Contribution chart
                fig1 = self.create_contribution_chart(df)
                if fig1:
                    st.plotly_chart(fig1)
                
                # Status chart
                fig2 = self.create_status_chart(df)
                if fig2:
                    st.plotly_chart(fig2)

if __name__ == '__main__':
    dashboard = ContributionDashboard()
    dashboard.run() 