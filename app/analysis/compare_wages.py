import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def load_and_clean_contribution_data(csv_path):
    """Load and clean contribution details from CSV."""
    print(f"\nLoading contribution details from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter valid entries and clean data
    valid_entries = df[df['CREDIT_MONTH'].str.contains(r'\d+/\d+', na=False)].copy()
    
    # Clean wages data
    valid_entries['WAGES'] = pd.to_numeric(valid_entries['WAGES'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    
    # Convert credit month to datetime for sorting
    valid_entries['DATE'] = pd.to_datetime(valid_entries['CREDIT_MONTH'].str.extract(r'(\d+/\d+)')[0], format='%m/%Y')
    
    # Sort by date
    valid_entries = valid_entries.sort_values('DATE')
    
    print(f"Found {len(valid_entries)} valid contribution records")
    return valid_entries

def load_and_clean_excel_data(excel_path):
    """Load and clean data from Excel file."""
    print(f"\nLoading Excel data from: {excel_path}")
    
    # Try to read from different sheets
    excel = pd.ExcelFile(excel_path)
    print(f"Available sheets: {excel.sheet_names}")
    
    df = None
    date_col = None
    wages_col = None
    
    # Try each sheet until we find valid data
    for sheet in excel.sheet_names:
        try:
            temp_df = pd.read_excel(excel_path, sheet_name=sheet)
            print(f"\nChecking sheet: {sheet}")
            print(f"Columns: {temp_df.columns.tolist()}")
            
            # Try to identify date and wages columns
            date_candidates = [col for col in temp_df.columns if any(key in str(col).upper() for key in ['MONTH', 'DATE'])]
            wage_candidates = [col for col in temp_df.columns if 'WAGE' in str(col).upper()]
            
            if date_candidates and wage_candidates:
                print(f"Found potential date column: {date_candidates[0]}")
                print(f"Found potential wages column: {wage_candidates[0]}")
                
                # Try to convert date column
                try:
                    # First try direct datetime conversion
                    test_date = pd.to_datetime(temp_df[date_candidates[0]], errors='coerce')
                    if not test_date.isna().all():
                        df = temp_df
                        date_col = date_candidates[0]
                        wages_col = wage_candidates[0]
                        print(f"Successfully found valid data in sheet: {sheet}")
                        break
                except:
                    continue
        except Exception as e:
            print(f"Error reading sheet {sheet}: {str(e)}")
            continue
    
    if df is None:
        raise ValueError("Could not find valid data in any sheet")
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame()
    
    # Convert date to same format as contribution details
    try:
        cleaned_df['DATE'] = pd.to_datetime(df[date_col])
    except:
        # If direct conversion fails, try to extract month and year
        try:
            # Try to extract month and year from the date column
            date_str = df[date_col].astype(str)
            month_year = date_str.str.extract(r'(\d{1,2})[/-](\d{4})')
            if not month_year.isna().all().all():
                cleaned_df['DATE'] = pd.to_datetime(month_year[0] + '/' + month_year[1], format='%m/%Y')
            else:
                raise ValueError("Could not parse dates")
        except Exception as e:
            print(f"Error parsing dates: {str(e)}")
            raise
    
    cleaned_df['CREDIT_MONTH'] = cleaned_df['DATE'].dt.strftime('%m/%Y')
    
    # Clean wages data
    cleaned_df['WAGES'] = pd.to_numeric(df[wages_col].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    
    # Remove rows with invalid wages
    cleaned_df = cleaned_df.dropna(subset=['WAGES'])
    cleaned_df = cleaned_df[cleaned_df['WAGES'] > 0]
    
    # Sort by date
    cleaned_df = cleaned_df.sort_values('DATE')
    
    print(f"Found {len(cleaned_df)} valid Excel records")
    print("\nSample data:")
    print(cleaned_df.head().to_string())
    
    return cleaned_df

def compare_wages(contrib_df, excel_df, name):
    """Compare wages between contribution details and Excel data."""
    print(f"\nComparing wages with {name}...")
    
    # Merge dataframes on date
    merged_df = pd.merge(
        contrib_df[['DATE', 'CREDIT_MONTH', 'WAGES', 'Status']],
        excel_df[['DATE', 'WAGES']],
        on='DATE',
        how='outer',
        suffixes=('_contrib', '_excel')
    )
    
    # Sort by date
    merged_df = merged_df.sort_values('DATE')
    
    # Calculate differences
    merged_df['DIFFERENCE'] = merged_df['WAGES_contrib'] - merged_df['WAGES_excel']
    merged_df['MISMATCH'] = abs(merged_df['DIFFERENCE']) > 1  # Allow for small rounding differences
    
    return merged_df

def plot_comparison(merged_df, name):
    """Create comparison plots."""
    # Line plot of wages over time
    fig1 = go.Figure()
    
    # Add contribution details line
    fig1.add_trace(go.Scatter(
        x=merged_df['DATE'],
        y=merged_df['WAGES_contrib'],
        name='Contribution Details',
        mode='lines+markers',
        hovertemplate='Date: %{x|%B %Y}<br>Wages: ₹%{y:,.2f}<extra></extra>'
    ))
    
    # Add Excel data line
    fig1.add_trace(go.Scatter(
        x=merged_df['DATE'],
        y=merged_df['WAGES_excel'],
        name='Excel Data',
        mode='lines+markers',
        hovertemplate='Date: %{x|%B %Y}<br>Wages: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig1.update_layout(
        title=f'Wages Comparison Over Time ({name})',
        xaxis_title='Date',
        yaxis_title='Wages (₹)',
        hovermode='x unified'
    )
    
    # Save plot
    fig1.write_html(f'wages_comparison_{name.lower().replace(" ", "_")}.html')
    
    # Create difference plot
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=merged_df['DATE'],
        y=merged_df['DIFFERENCE'],
        name='Difference',
        hovertemplate='Date: %{x|%B %Y}<br>Difference: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig2.update_layout(
        title=f'Wage Differences ({name}) - Contribution minus Excel',
        xaxis_title='Date',
        yaxis_title='Difference (₹)',
        hovermode='x unified'
    )
    
    # Save plot
    fig2.write_html(f'wages_difference_{name.lower().replace(" ", "_")}.html')

def analyze_comparison(merged_df, name):
    """Analyze and print comparison results."""
    print(f"\nComparison Summary for {name}:")
    print(f"Total records: {len(merged_df)}")
    
    # Count valid comparisons (where we have both values)
    valid_comparisons = merged_df.dropna(subset=['WAGES_contrib', 'WAGES_excel'])
    print(f"Valid comparisons: {len(valid_comparisons)}")
    
    if len(valid_comparisons) > 0:
        mismatches = valid_comparisons[valid_comparisons['MISMATCH']]
        print(f"Mismatches: {len(mismatches)}")
        print(f"Match percentage: {((len(valid_comparisons) - len(mismatches)) / len(valid_comparisons) * 100):.1f}%")
        
        if len(mismatches) > 0:
            print("\nMismatched Records:")
            mismatches = mismatches.copy()
            mismatches['DATE'] = mismatches['DATE'].dt.strftime('%m/%Y')
            print(mismatches[['DATE', 'WAGES_contrib', 'WAGES_excel', 'DIFFERENCE', 'Status']].to_string())
    
    # Save comparison results
    output_file = f'wages_comparison_{name.lower().replace(" ", "_")}.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved comparison results to: {output_file}")

def main():
    # File paths
    contrib_path = "contribution_details_20250210_202555.csv"
    excel_files = [
        ("2101-312/DEBENDRA NATH AGASTI.xlsx", "Personal File"),
        ("2101-312/MAIN_FILE_APR-2024.xlsx", "Main File")
    ]
    
    if not Path(contrib_path).exists():
        print(f"Contribution details file not found: {contrib_path}")
        return
    
    try:
        # Load contribution data
        contrib_df = load_and_clean_contribution_data(contrib_path)
        
        # Compare with each Excel file
        for excel_path, name in excel_files:
            if not Path(excel_path).exists():
                print(f"\nExcel file not found: {excel_path}")
                continue
            
            try:
                print(f"\n{'='*40}")
                print(f"Processing {name}")
                print('='*40)
                
                # Load Excel data
                excel_df = load_and_clean_excel_data(excel_path)
                
                # Compare wages
                merged_df = compare_wages(contrib_df, excel_df, name)
                
                # Analyze results
                analyze_comparison(merged_df, name)
                
                # Create plots
                plot_comparison(merged_df, name)
                
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
        
    except Exception as e:
        print(f"Error comparing wages: {str(e)}")

if __name__ == "__main__":
    main() 