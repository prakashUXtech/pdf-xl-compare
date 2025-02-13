"""
UI component for file comparison view.
"""
import streamlit as st
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pathlib import Path
from typing import Optional, Tuple
from app.services.ocr.nanonets import NanonetsOCR
from app.processors.file_processor import FileProcessor
from app.services.comparison import ComparisonService
from app.services.ocr.nanonets_extract import clean_saved_response
import json
from io import BytesIO
import os
import glob

class ComparisonView:
    """UI component for file comparison."""
    
    def __init__(self, ocr_service: NanonetsOCR):
        """Initialize with OCR service."""
        self.ocr_service = ocr_service
        self.file_processor = FileProcessor()
        self.comparison_service = ComparisonService()
        
        # Initialize session state for progress tracking
        if 'progress_status' not in st.session_state:
            st.session_state.progress_status = {
                'pdf_uploaded': False,
                'pdf_processed': False,
                'excel_uploaded': False,
                'comparison_done': False
            }
    
    def _show_progress_tracker(self):
        """Display progress tracker in sidebar."""
        with st.sidebar:
            st.markdown("### Progress Tracker")
            
            # PDF Upload Status
            status_icon = "✅" if st.session_state.progress_status['pdf_uploaded'] else "⬜"
            st.markdown(f"{status_icon} 1. Upload PDF")
            
            # PDF Processing Status
            status_icon = "✅" if st.session_state.progress_status['pdf_processed'] else "⬜"
            st.markdown(f"{status_icon} 2. Process PDF Data")
            
            # Excel Upload Status
            status_icon = "✅" if st.session_state.progress_status['excel_uploaded'] else "⬜"
            st.markdown(f"{status_icon} 3. Upload Excel")
            
            # Comparison Status
            status_icon = "✅" if st.session_state.progress_status['comparison_done'] else "⬜"
            st.markdown(f"{status_icon} 4. Compare Files")
            
            st.markdown("---")
            if all(st.session_state.progress_status.values()):
                st.success("✨ All steps completed!")
    
    def show_raw_response_tab(self):
        """Display the raw response handling and PDF processing tab UI."""
        # Show progress tracker in sidebar
        self._show_progress_tracker()
        
        st.header("Process PDF & Raw Response")
        
        # PDF Upload Section
        st.markdown("""
        ### Upload PDF
        Upload your PDF file containing wage data. The system will automatically extract and process the wage information.
        """)
        
        pdf_file = st.file_uploader(
            "Choose a PDF file", 
            type=['pdf'],
            key="pdf_upload_raw",
            help="Upload a PDF file containing wage information. The file will be automatically processed."
        )
        
        if pdf_file:
            st.session_state.progress_status['pdf_uploaded'] = True
            pdf_path = Path("data/uploads") / pdf_file.name
            
            try:
                with st.spinner("Processing PDF file..."):
                    # Save PDF file
                    pdf_content = pdf_file.getvalue()
                    self.file_processor.save_temp_file(pdf_content, pdf_path)
                    
                    # Extract data using OCR service
                    st.info("🔍 Extracting data from PDF...")
                    response = self.ocr_service.extract_tables(pdf_path)
                    
                    if response is not None:
                        # Get the latest raw response file
                        raw_responses_dir = Path("raw_responses")
                        if raw_responses_dir.exists():
                            response_files = list(raw_responses_dir.glob("*.json"))
                            if response_files:
                                latest_file = max(response_files, key=lambda x: x.stat().st_mtime)
                                
                                # Process the raw response directly
                                with st.spinner("Cleaning and processing extracted data..."):
                                    cleaned_data = clean_saved_response(str(latest_file))
                                    
                                    if cleaned_data is not None and not cleaned_data.empty:
                                        st.session_state.progress_status['pdf_processed'] = True
                                        st.success(f"""
                                        ✅ PDF processed successfully
                                        - Data extracted and cleaned
                                        - Total rows: {len(cleaned_data)}
                                        - Columns: {', '.join(cleaned_data.columns)}
                                        """)
                                        
                                        # Auto-save the cleaned data
                                        self._auto_save_cleaned_data(cleaned_data, latest_file.name)
                                        
                                        # Show preview in an expander
                                        with st.expander("Preview Processed Data", expanded=True):
                                            st.dataframe(cleaned_data.head(), use_container_width=True)
                                    else:
                                        st.error("❌ Could not clean the extracted data.")
                            else:
                                st.error("❌ No raw response file found after processing.")
                        else:
                            st.error("❌ Raw responses directory not found.")
                    else:
                        st.error("❌ Could not extract data from PDF. Please ensure the file contains wage information in a table format.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.session_state.progress_status['pdf_processed'] = False
            finally:
                self.file_processor.cleanup_temp_files(pdf_path)
        
        # Keep the manual processing section in an expander for advanced users
        with st.expander("Advanced: Manual Raw Response Processing", expanded=False):
            st.markdown("""
            This section allows manual processing of saved raw responses. 
            Use this only if you need to reprocess previously saved response files.
            """)
            
            # Use Saved Response section with improved organization
            raw_responses_dir = Path("raw_responses")
            if raw_responses_dir.exists():
                response_files = list(raw_responses_dir.glob("*.json"))
                if response_files:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        selected_file = st.selectbox(
                            "Select saved response file",
                            response_files,
                            format_func=lambda x: x.name,
                            help="Choose a previously saved raw response file to process"
                        )
                    
                    with col2:
                        if selected_file:
                            if st.button(
                                "Process Selected File",
                                help="Clean and process the selected raw response file",
                                use_container_width=True
                            ):
                                with st.spinner("Processing response file..."):
                                    cleaned_data = clean_saved_response(str(selected_file))
                                    if cleaned_data is not None and not cleaned_data.empty:
                                        st.success(f"""
                                        ✅ Successfully cleaned response
                                        - Rows processed: {len(cleaned_data)}
                                        - Data ready for comparison
                                        """)
                                        self._auto_save_cleaned_data(cleaned_data, selected_file.name)
                else:
                    st.info("ℹ️ No saved responses found.")
            else:
                st.warning("⚠️ Raw responses directory not found.")
    
    def show_processed_csv_tab(self):
        """Display the processed files comparison tab UI."""
        # Show progress tracker in sidebar
        self._show_progress_tracker()
        
        st.header("Compare with Processed Files")
        
        # List processed files
        st.markdown("""
        ### Step 3: Select Processed File
        Choose a processed file to compare with your Excel data.
        """)
        
        processed_dir = Path("data/processed")
        selected_processed_data = None  # Initialize variable to store selected file data
        
        if processed_dir.exists():
            processed_files = list(processed_dir.glob("cleaned_*.csv"))
            if processed_files:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_processed_file = st.selectbox(
                        "Select processed file",
                        processed_files,
                        format_func=lambda x: x.name,
                        help="These are the cleaned and processed files from previous operations"
                    )
                
                if selected_processed_file:
                    try:
                        with st.spinner("Loading processed file..."):
                            selected_processed_data = pd.read_csv(selected_processed_file)
                            
                            with col2:
                                st.metric(
                                    "Rows in File",
                                    len(selected_processed_data),
                                    help="Total number of wage entries in the processed file"
                                )
                            
                            st.success(f"✅ Loaded: {selected_processed_file.name}")
                            
                            with st.expander("Preview processed data", expanded=False):
                                st.dataframe(selected_processed_data.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
                        selected_processed_data = None
            else:
                st.info("ℹ️ No processed files found. Please process some files first.")
        else:
            st.warning("⚠️ Processed files directory not found")
        
        # Excel Upload Section
        st.markdown("---")
        st.markdown("""
        ### Step 4: Upload Excel for Comparison
        Upload your Excel file to compare with the processed data. The system will automatically identify and highlight any mismatches.
        """)
        
        if selected_processed_data is None:
            st.warning("⚠️ Please select a processed file first before uploading Excel")
            return
            
        excel_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            key="excel_upload_comparison",
            help="Upload an Excel file containing wage data to compare with the processed file"
        )
        
        if excel_file:
            st.session_state.progress_status['excel_uploaded'] = True
            excel_path = Path("data/uploads") / excel_file.name
            try:
                with st.spinner("Processing Excel file..."):
                    excel_content = excel_file.getvalue()
                    self.file_processor.save_temp_file(excel_content, excel_path)
                    excel_data = self.file_processor.load_excel_data(excel_path)
                    
                    if excel_data is not None and not excel_data.empty:
                        st.success(f"""
                        ✅ Excel file loaded successfully
                        - File: {excel_file.name}
                        - Rows: {len(excel_data)}
                        """)
                        
                        # Store Excel content and update progress
                        self._current_excel_content = excel_content
                        st.session_state['comparison_df1'] = selected_processed_data
                        st.session_state['comparison_df2'] = excel_data
                        st.session_state['comparison_excel_filename'] = excel_file.name
                        
                        # Show comparison results
                        st.markdown("---")
                        st.markdown("### Comparison Results")
                        self._compare_and_show_results(
                            df1=selected_processed_data,
                            df2=excel_data,
                            excel_filename=excel_file.name
                        )
                        
                        # Update progress status
                        st.session_state.progress_status['comparison_done'] = True
                    else:
                        st.error("❌ No data found in Excel file.")
            except Exception as e:
                st.error(f"❌ Error processing Excel file: {str(e)}")
            finally:
                self.file_processor.cleanup_temp_files(excel_path)
    
    def _auto_save_cleaned_data(self, cleaned_data: pd.DataFrame, source_filename: str):
        """Automatically save cleaned data to both CSV and Excel formats."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate base filename from source
            base_filename = f"cleaned_{Path(source_filename).stem}"
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as CSV
            csv_file = output_dir / f"{base_filename}_{timestamp}.csv"
            cleaned_data.to_csv(csv_file, index=False)
            
            # Save as Excel
            excel_file = output_dir / f"{base_filename}_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                cleaned_data.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            # Store last saved files in session state
            if 'last_saved_files' not in st.session_state:
                st.session_state.last_saved_files = []
            
            st.session_state.last_saved_files.extend([
                str(csv_file.absolute()),
                str(excel_file.absolute())
            ])
            
            # Show success message
            st.success(f"""
            ✅ Data automatically saved:
            - CSV: {csv_file.name}
            - Excel: {excel_file.name}
            
            Location: {output_dir.absolute()}
            Rows saved: {len(cleaned_data)}
            """)
            
            # Print to console for debugging
            print(f"Auto-saved files at {timestamp}:")
            print(f"CSV: {csv_file.absolute()}")
            print(f"Excel: {excel_file.absolute()}")
            
        except Exception as e:
            st.error(f"Error during auto-save: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            print(f"Auto-save error: {str(e)}")
            print(traceback.format_exc())
    
    def _handle_cleaned_data(self, cleaned_data: pd.DataFrame):
        """Handle cleaned data from raw response."""
        # Initialize session state for save status if not exists
        if 'save_status' not in st.session_state:
            st.session_state.save_status = []
        if 'last_saved_file' not in st.session_state:
            st.session_state.last_saved_file = None
        
        # Show any persisted save status messages
        for msg in st.session_state.save_status:
            if msg['type'] == 'success':
                st.success(msg['text'])
            elif msg['type'] == 'error':
                st.error(msg['text'])
            elif msg['type'] == 'info':
                st.info(msg['text'])
            elif msg['type'] == 'warning':
                st.warning(msg['text'])
        
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_data.head(), use_container_width=True)
        
        # Save option
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user specify a filename
            default_filename = f"cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            filename = st.text_input("Filename (without extension)", value=default_filename)
            
            if not filename:
                filename = default_filename
            
            # Add file format selection
            file_format = st.selectbox(
                "Save as",
                options=["CSV", "Excel"],
                index=0
            )
        
        with col2:
            st.write("Preview of data to be saved:")
            st.write(f"Total rows: {len(cleaned_data)}")
            st.write(f"Columns: {', '.join(cleaned_data.columns)}")
            
            # Show last saved file info if exists
            if st.session_state.last_saved_file:
                st.info(f"Last saved file: {st.session_state.last_saved_file}")
        
        def add_status_message(msg_type: str, text: str):
            """Add a message to session state."""
            st.session_state.save_status.append({
                'type': msg_type,
                'text': text,
                'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
            })
        
        def clear_status_messages():
            """Clear all status messages."""
            st.session_state.save_status = []
        
        # Separate save and download buttons
        col3, col4 = st.columns(2)
        
        with col3:
            # Use a form to prevent page refresh
            with st.form(key='save_form'):
                save_button = st.form_submit_button("Save to Disk")
                if save_button:
                    try:
                        # Clear previous status messages
                        clear_status_messages()
                        
                        add_status_message('info', "Starting save operation...")
                        
                        # Create output directory if it doesn't exist
                        output_dir = Path("data/processed")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        add_status_message('info', f"Output directory ready: {output_dir.absolute()}")
                        
                        # Log to console for debugging
                        print(f"Save operation started at {pd.Timestamp.now()}")
                        print(f"Output directory: {output_dir.absolute()}")
                        
                        # Ensure DataFrame is not empty
                        if cleaned_data.empty:
                            add_status_message('error', "No data to save!")
                            return
                        
                        add_status_message('info', f"Preparing to save {len(cleaned_data)} rows of data...")
                        
                        if file_format == "CSV":
                            output_file = output_dir / f"{filename}.csv"
                            print(f"Saving CSV to: {output_file.absolute()}")
                            add_status_message('info', f"Saving as CSV to: {output_file}")
                            
                            cleaned_data.to_csv(output_file, index=False)
                            
                            if output_file.exists():
                                file_size = output_file.stat().st_size / 1024  # KB
                                success_msg = f"""
                                ✅ File saved successfully!
                                - Path: {output_file.absolute()}
                                - Size: {file_size:.1f} KB
                                - Rows: {len(cleaned_data)}
                                - Format: CSV
                                """
                                add_status_message('success', success_msg)
                                print(success_msg)
                                
                                # Store last saved file
                                st.session_state.last_saved_file = str(output_file.absolute())
                                
                                # Verify the saved file
                                try:
                                    test_read = pd.read_csv(output_file)
                                    add_status_message('info', f"Verified file is readable with {len(test_read)} rows")
                                except Exception as verify_error:
                                    add_status_message('warning', f"File saved but verification failed: {str(verify_error)}")
                            else:
                                add_status_message('error', "Failed to create file!")
                                
                        else:  # Excel
                            output_file = output_dir / f"{filename}.xlsx"
                            print(f"Saving Excel to: {output_file.absolute()}")
                            add_status_message('info', f"Saving as Excel to: {output_file}")
                            
                            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                                cleaned_data.to_excel(writer, index=False, sheet_name='Cleaned Data')
                            
                            if output_file.exists():
                                file_size = output_file.stat().st_size / 1024  # KB
                                success_msg = f"""
                                ✅ File saved successfully!
                                - Path: {output_file.absolute()}
                                - Size: {file_size:.1f} KB
                                - Rows: {len(cleaned_data)}
                                - Format: Excel
                                """
                                add_status_message('success', success_msg)
                                print(success_msg)
                                
                                # Store last saved file
                                st.session_state.last_saved_file = str(output_file.absolute())
                                
                                # Verify the saved file
                                try:
                                    test_read = pd.read_excel(output_file)
                                    add_status_message('info', f"Verified file is readable with {len(test_read)} rows")
                                except Exception as verify_error:
                                    add_status_message('warning', f"File saved but verification failed: {str(verify_error)}")
                            else:
                                add_status_message('error', "Failed to create file!")
                        
                    except Exception as e:
                        error_msg = f"Error saving to disk: {str(e)}"
                        add_status_message('error', error_msg)
                        print(error_msg)
                        
                        import traceback
                        trace_msg = f"Detailed error: {traceback.format_exc()}"
                        add_status_message('error', trace_msg)
                        print(trace_msg)
                        
                        # List directory contents for debugging
                        try:
                            dir_contents = list(output_dir.glob('*'))
                            add_status_message('info', f"Directory contents: {dir_contents}")
                            print(f"Directory contents: {dir_contents}")
                        except Exception as dir_error:
                            add_status_message('error', f"Error listing directory: {str(dir_error)}")
        
        with col4:
            try:
                if file_format == "CSV":
                    # Prepare CSV data
                    csv_data = cleaned_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download File",
                        data=csv_data,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    # Prepare Excel data
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        cleaned_data.to_excel(writer, index=False)
                    excel_data = output.getvalue()
                    st.download_button(
                        "Download File",
                        data=excel_data,
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")
        
        # Add clear messages button
        if st.session_state.save_status:
            if st.button("Clear Messages"):
                clear_status_messages()
        
        # Compare with Excel section
        st.subheader("Compare with Excel")
        excel_file = st.file_uploader(
            "Upload Excel file for comparison",
            type=['xlsx', 'xls'],
            key="excel_upload_cleaned"
        )
        
        if excel_file:
            self._handle_pdf_excel_comparison(selected_processed_data, excel_file)
    
    def _handle_pdf_excel_comparison(self, pdf_data: pd.DataFrame, excel_file: UploadedFile):
        """Handle comparison between PDF and Excel data."""
        excel_path = Path("temp.xlsx")
        try:
            # Store the original content
            self._current_excel_content = excel_file.getvalue()
            
            self.file_processor.save_temp_file(self._current_excel_content, excel_path)
            excel_data = self.file_processor.load_excel_data(excel_path)
            
            if excel_data is not None and not excel_data.empty:
                # Compare PDF data with Excel
                self._compare_and_show_results(pdf_data, excel_data, excel_file.name)
            else:
                st.error("No data found in Excel file.")
        finally:
            self.file_processor.cleanup_temp_files(excel_path)
    
    def _compare_and_show_results(self, df1: pd.DataFrame, df2: pd.DataFrame, excel_filename: str = None):
        """Compare and display results with improved organization and feedback."""
        # Initialize session state
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = None
            st.session_state.excel_filename = None
            st.session_state.current_excel_content = None

        # Compute comparison if not already done or if "Recompare Files" is clicked
        compute_comparison = False
        if st.session_state.comparison_results is None:
            compute_comparison = True
        elif st.button("🔄 Recompare Files", help="Run the comparison again with the current files"):
            compute_comparison = True

        if compute_comparison:
            with st.spinner("Comparing files..."):
                comparison = self.comparison_service.compare_data(df1, df2)
                st.session_state.comparison_results = comparison
                st.session_state.excel_filename = excel_filename
                st.session_state.current_excel_content = self._current_excel_content

        # Use stored comparison results
        comparison = st.session_state.comparison_results
        if comparison is not None and not comparison.empty:
            # Show comparison summary
            mismatches = comparison[comparison['MISMATCH']]
            total_records = len(comparison)
            mismatch_count = len(mismatches)
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Records",
                    f"{total_records:,}",
                    help="Total number of records compared"
                )
            with col2:
                st.metric(
                    "Mismatches",
                    f"{mismatch_count:,}",
                    delta=f"{-mismatch_count}" if mismatch_count > 0 else None,
                    delta_color="inverse",
                    help="Number of records with wage mismatches"
                )
            with col3:
                if not mismatches.empty:
                    total_diff = abs(mismatches['DIFFERENCE']).sum()
                    st.metric(
                        "Total Difference",
                        f"₹{total_diff:,.2f}",
                        help="Sum of all wage differences"
                    )
            with col4:
                if not mismatches.empty:
                    avg_diff = abs(mismatches['DIFFERENCE']).mean()
                    st.metric(
                        "Average Difference",
                        f"₹{avg_diff:,.2f}",
                        help="Average wage difference per mismatch"
                    )
            
            # Show status message based on results
            if mismatches.empty:
                st.success("✅ Perfect Match! All wages match between PDF and Excel.")
            else:
                st.error(f"⚠️ Found {mismatch_count:,} mismatches out of {total_records:,} records.")
            
            # Download Options
            st.markdown("### Download Options")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.markdown("##### Comparison Report")
                # Download button for comparison CSV
                st.download_button(
                    "📊 Download Full Comparison (CSV)",
                    comparison.drop(columns=['MISMATCH']).to_csv(index=False),
                    f"{Path(st.session_state.excel_filename).stem}_comparison.csv" if st.session_state.excel_filename else "comparison.csv",
                    "text/csv",
                    help="Download the complete comparison including all records"
                )
            
            with download_col2:
                st.markdown("##### Modified Excel Files")
                # Get the original column names from df2 (Excel data)
                month_col = next((col for col in df2.columns if 'MONTH' in col), None)
                wages_col = next((col for col in df2.columns if 'WAGE' in col), None)
                
                if month_col and wages_col and st.session_state.current_excel_content is not None:
                    try:
                        # First button for highlighted Excel
                        highlighted_excel = self.file_processor.highlight_mismatches_in_original_excel(
                            st.session_state.current_excel_content,
                            comparison,
                            month_col,
                            wages_col
                        )
                        st.download_button(
                            "🔍 Download Highlighted Excel",
                            highlighted_excel,
                            f"{Path(st.session_state.excel_filename).stem}_highlighted.xlsx" if st.session_state.excel_filename else "highlighted_comparison.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Original Excel file with mismatched cells highlighted"
                        )
                        
                        # Second button for fixed and highlighted Excel
                        fixed_highlighted_excel = self.file_processor.fix_and_highlight_excel(
                            st.session_state.current_excel_content,
                            comparison,
                            month_col,
                            wages_col
                        )
                        st.download_button(
                            "✨ Download Fixed & Highlighted Excel",
                            fixed_highlighted_excel,
                            f"{Path(st.session_state.excel_filename).stem}_fixed_highlighted.xlsx" if st.session_state.excel_filename else "fixed_highlighted_comparison.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Excel file with mismatched cells corrected using PDF values and highlighted"
                        )
                    except Exception as e:
                        st.error(f"❌ Could not create Excel files: {str(e)}")
            
            # Detailed Results
            st.markdown("### Detailed Results")
            
            # Show mismatches first if they exist
            if not mismatches.empty:
                with st.expander("View Mismatches", expanded=True):
                    st.markdown(f"##### Showing {len(mismatches)} mismatched records:")
                    
                    # Format the display data
                    display_df = mismatches.copy()
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
            
            # Show all records in an expander
            with st.expander("View All Records", expanded=False):
                st.markdown(f"##### Showing all {len(comparison)} records:")
                st.dataframe(
                    comparison.drop(columns=['MISMATCH']),
                    use_container_width=True,
                    hide_index=True
                )
    
    @staticmethod
    def _load_raw_responses() -> dict:
        """Load raw responses from files."""
        raw_responses_dir = Path("raw_responses")
        if not raw_responses_dir.exists():
            return {}
            
        raw_responses = {}
        for file in raw_responses_dir.glob("*.json"):
            with open(file, 'r') as f:
                raw_responses[file.stem] = json.load(f)
        return raw_responses
    
    @staticmethod
    def _find_matching_response(pdf_hash: str, raw_responses: dict) -> Optional[dict]:
        """Find a matching raw response for the given PDF hash."""
        for response_data in raw_responses.values():
            if isinstance(response_data, dict) and response_data.get('pdf_hash') == pdf_hash:
                return response_data
        return None 