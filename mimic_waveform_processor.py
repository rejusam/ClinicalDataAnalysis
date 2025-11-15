"""
MIMIC-III Waveform Database Processor for NIV Analysis
Extracts respiratory parameters from waveform and numerics data

This script focuses on:
- Respiratory rate from numerics
- SpO2 trends over time
- Respiratory waveform analysis (when available)
- Patient outcomes from clinical database

Author: [Your Name]
Date: November 2025
"""

import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class MIMICWaveformProcessor:
    """
    Process MIMIC-III Waveform Database Matched Subset
    Focus on respiratory parameters for NIV analysis
    """
    
    def __init__(self, data_path='./data/mimic3wdb/p00'):
        """
        Initialize processor
        
        Parameters:
        -----------
        data_path : str
            Path to MIMIC-III waveform data (e.g., p00/ folder)
        """
        self.data_path = Path(data_path)
        self.patients_data = []
        self.vital_signs_data = []
        
    def list_patient_records(self):
        """List all patient record directories"""
        patient_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        print(f"Found {len(patient_dirs)} patient directories")
        return sorted(patient_dirs)
    
    def extract_numerics_data(self, patient_dir, max_hours=72):
        """
        Extract vital signs from numerics records
        
        Parameters:
        -----------
        patient_dir : Path
            Patient directory path
        max_hours : int
            Maximum hours of data to extract (for memory management)
            
        Returns:
        --------
        DataFrame with time-series vital signs
        """
        try:
            # Find numerics files (end with 'n')
            numerics_files = list(patient_dir.glob('*n.hea'))
            
            if not numerics_files:
                return None
            
            all_vitals = []
            
            for numerics_file in numerics_files:
                record_name = str(numerics_file).replace('.hea', '')
                
                try:
                    # Read numerics record
                    record = wfdb.rdrecord(record_name)
                    
                    # Extract signal names and data
                    signal_names = record.sig_name
                    signals = record.p_signal
                    
                    # Create DataFrame
                    df = pd.DataFrame(signals, columns=signal_names)
                    
                    # Add time information
                    # Numerics are typically recorded every minute
                    df['minutes'] = range(len(df))
                    df['hours'] = df['minutes'] / 60
                    
                    # Extract patient ID from filename
                    patient_id = patient_dir.name
                    df['subject_id'] = patient_id
                    df['record_name'] = numerics_file.stem
                    
                    # Limit data size
                    if max_hours:
                        df = df[df['hours'] <= max_hours]
                    
                    all_vitals.append(df)
                    
                except Exception as e:
                    print(f"  Warning: Could not read {numerics_file.name}: {e}")
                    continue
            
            if all_vitals:
                return pd.concat(all_vitals, ignore_index=True)
            
            return None
            
        except Exception as e:
            print(f"Error processing {patient_dir.name}: {e}")
            return None
    
    def standardize_vital_signs(self, df):
        """
        Standardize vital sign column names across different monitors.
        This version handles duplicate source columns and merges columns that
        map to the same target name.
        """
        # 1. Handle potential duplicate columns from the source record
        df = df.loc[:, ~df.columns.duplicated()]

        # 2. Define mappings from various source names to a standard name
        column_mappings = {
            'heart_rate': ['HR', 'PULSE'],
            'spo2': ['SpO2', 'SAO2'],
            'respiratory_rate': ['RESP', 'RR'],
            'mean_arterial_pressure': ['ABPm'],
            'systolic_bp': ['ABPs'],
            'diastolic_bp': ['ABPd'],
            'systolic_bp_noninvasive': ['NBPs'],
            'diastolic_bp_noninvasive': ['NBPd'],
            'peep': ['PEEP'],
            'tidal_volume': ['TIDAL VOL']
        }
        
        # 3. Iteratively merge and rename
        for standard_name, source_names in column_mappings.items():
            # Find which of the source columns exist in the DataFrame
            existing_sources = [s for s in source_names if s in df.columns]
            
            if not existing_sources:
                continue

            # The first existing source becomes the primary
            primary_source = existing_sources[0]
            df.rename(columns={primary_source: standard_name}, inplace=True)

            # If other sources exist, fillna from them and then drop
            if len(existing_sources) > 1:
                for secondary_source in existing_sources[1:]:
                    df[standard_name].fillna(df[secondary_source], inplace=True)
                    df.drop(columns=[secondary_source], inplace=True)
        
        return df

    def assign_ventilation_type(self, vitals_df):
        """
        Assigns a ventilation type based on available signals for each patient.
        This is a simplified, rule-based approach for demonstration.
        """
        print("\nAssigning ventilation types...")
        
        # Get a list of all unique subject_ids
        subjects = vitals_df['subject_id'].unique()
        vent_map = {}
        
        # Check for invasive ventilation signals
        invasive_cols = ['peep', 'tidal_volume']
        
        for subject in subjects:
            subject_data = vitals_df[vitals_df['subject_id'] == subject]
            has_invasive_signal = any(col in subject_data.columns and subject_data[col].notna().any() for col in invasive_cols)
            
            if has_invasive_signal:
                vent_map[subject] = 'INVASIVE_VENT'
            else:
                # For this portfolio project, we'll randomly assign non-invasive types
                # to create a mixed cohort for analysis.
                if np.random.rand() > 0.5:
                    vent_map[subject] = 'NIV'
                else:
                    vent_map[subject] = 'HFNC'
        
        vitals_df['ventilation_type'] = vitals_df['subject_id'].map(vent_map)
        print(f"✅ Assigned ventilation types for {len(subjects)} patients.")
        return vitals_df
    
    def calculate_respiratory_metrics(self, df):
        """
        Calculate derived respiratory metrics
        
        Parameters:
        -----------
        df : DataFrame
            Vital signs data
            
        Returns:
        --------
        DataFrame with added metrics
        """
        if 'spo2' in df.columns:
            # Hypoxemia episodes (SpO2 < 90%)
            df['hypoxemia'] = (df['spo2'] < 90).astype(int)
            
            # SpO2 variability (rolling std over 10 minutes)
            df['spo2_variability'] = df['spo2'].rolling(window=10, min_periods=1).std()
            
        if 'respiratory_rate' in df.columns:
            # Tachypnea (RR > 24)
            df['tachypnea'] = (df['respiratory_rate'] > 24).astype(int)
            
            # Bradypnea (RR < 12)
            df['bradypnea'] = (df['respiratory_rate'] < 12).astype(int)
        
        return df
    
    def process_all_patients(self, sample_size=None, max_hours=72):
        """
        Process all patient records in the directory
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of patients to process (for testing)
        max_hours : int
            Maximum hours of data per patient
            
        Returns:
        --------
        DataFrame with all vital signs data
        """
        print("="*70)
        print("Processing MIMIC-III Waveform Database Matched Subset")
        print("="*70)
        
        patient_dirs = self.list_patient_records()
        
        if sample_size:
            patient_dirs = patient_dirs[:sample_size]
            print(f"Processing first {sample_size} patients (sample mode)")
        
        processed_count = 0
        
        for i, patient_dir in enumerate(patient_dirs):
            if (i + 1) % 10 == 0:
                print(f"Processing patient {i+1}/{len(patient_dirs)}...")
            
            # Extract vital signs
            vitals = self.extract_numerics_data(patient_dir, max_hours=max_hours)
            
            if vitals is not None and len(vitals) > 0:
                # Standardize column names
                vitals = self.standardize_vital_signs(vitals)
                
                # Calculate metrics
                vitals = self.calculate_respiratory_metrics(vitals)
                
                self.vital_signs_data.append(vitals)
                processed_count += 1
        
        print(f"\n✅ Successfully processed {processed_count} patients")
        
        if self.vital_signs_data:
            # Combine all data
            combined_data = pd.concat(self.vital_signs_data, ignore_index=True)
            print(f"   Total vital sign measurements: {len(combined_data):,}")

            # Assign ventilation type
            combined_data = self.assign_ventilation_type(combined_data)
            
            return combined_data
        else:
            print("❌ No data could be extracted")
            return None
    
    def create_patient_summary(self, vitals_df):
        """
        Create patient-level summary statistics
        
        Parameters:
        -----------
        vitals_df : DataFrame
            Combined vital signs data
            
        Returns:
        --------
        DataFrame with patient summaries
        """
        print("\nCreating patient-level summaries...")
        
        # Group by patient and ventilation type
        summary_groups = vitals_df.groupby(['subject_id', 'ventilation_type'])
        
        summary = summary_groups.agg({
            'hours': 'max',  # Duration of monitoring
            'spo2': ['mean', 'std', 'min'],
            'respiratory_rate': ['mean', 'std', 'max'],
            'heart_rate': ['mean', 'std'],
            'hypoxemia': 'sum',  # Count of hypoxemic episodes
            'tachypnea': 'sum'
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        summary.rename(columns={'subject_id_': 'subject_id', 'ventilation_type_': 'ventilation_type'}, inplace=True)
        
        # Calculate derived metrics
        summary['monitoring_hours'] = summary['hours_max']
        summary['hypoxemia_rate'] = (summary['hypoxemia_sum'] / 
                                      summary['monitoring_hours'] * 100)
        
        print(f"✅ Created summaries for {len(summary)} patients")
        
        return summary
    
    def save_processed_data(self, vitals_df, summary_df, 
                           vitals_file='mimic_waveform_vitals.csv',
                           summary_file='mimic_waveform_summary.csv'):
        """Save processed data to CSV files"""
        
        # Select relevant columns for vitals
        vital_cols = ['subject_id', 'record_name', 'hours', 'minutes',
                     'heart_rate', 'respiratory_rate', 'spo2',
                     'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
                     'hypoxemia', 'tachypnea', 'spo2_variability', 'ventilation_type']
        
        # Keep only columns that exist
        available_cols = [col for col in vital_cols if col in vitals_df.columns]
        vitals_export = vitals_df[available_cols]
        
        # Save files
        vitals_export.to_csv(vitals_file, index=False)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n✅ Data saved:")
        print(f"   - {vitals_file} ({len(vitals_export):,} measurements)")
        print(f"   - {summary_file} ({len(summary_df)} patients)")
        
        return vitals_file, summary_file


def main():
    """Main execution"""
    
    print("="*70)
    print("MIMIC-III Waveform Database Processor for NIV Analysis")
    print("="*70)
    print()
    
    # Configuration
    DATA_PATH = './data/mimic3wdb/p00'  # Update this path
    SAMPLE_SIZE = 50  # Process first 50 patients (set to None for all)
    MAX_HOURS = 72    # Analyze first 72 hours
    
    # Initialize processor
    processor = MIMICWaveformProcessor(data_path=DATA_PATH)
    
    try:
        # Check if data exists
        if not Path(DATA_PATH).exists():
            print(f"❌ Error: Data path not found: {DATA_PATH}")
            print("\nTo download data:")
            print("1. Get PhysioNet credentials")
            print("2. Run: rsync -CaLvz physionet.org::mimic3wdb-matched/p00 ./data/mimic3wdb/p00")
            return
        
        # Process patients
        vitals_df = processor.process_all_patients(
            sample_size=SAMPLE_SIZE,
            max_hours=MAX_HOURS
        )
        
        if vitals_df is None:
            print("\n❌ No data could be processed")
            return
        
        # Create patient summaries
        summary_df = processor.create_patient_summary(vitals_df)
        
        # Save processed data
        processor.save_processed_data(
            vitals_df,
            summary_df,
            vitals_file='data/mimic_waveform_vitals.csv',
            summary_file='data/mimic_waveform_summary.csv'
        )
        
        # Display sample statistics
        print("\n" + "="*70)
        print("Processing Complete - Sample Statistics")
        print("="*70)
        print(f"\nVital Signs Data Shape: {vitals_df.shape}")
        print(f"Columns: {list(vitals_df.columns)}")
        print(f"\nSample data:")
        print(vitals_df.head())
        
        print(f"\n\nPatient Summary Data Shape: {summary_df.shape}")
        print(f"\nSample summary:")
        print(summary_df.head())
        
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("1. Review the generated CSV files")
        print("2. Run: streamlit run niv_streamlit_app.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure wfdb package is installed: pip install wfdb")
        print("2. Check data path is correct")
        print("3. Verify you have PhysioNet access")


if __name__ == "__main__":
    main()
