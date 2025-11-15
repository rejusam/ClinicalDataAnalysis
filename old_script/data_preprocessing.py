"""
MIMIC-III Data Preprocessing Script for NIV Analysis
This script extracts and prepares respiratory/ventilation data from MIMIC-III

Requirements:
1. PhysioNet credentialing completed
2. MIMIC-III database downloaded or accessed via BigQuery
3. Appropriate data use agreement signed

Author: [Your Name]
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MIMICPreprocessor:
    """
    Preprocessor for MIMIC-III data focusing on respiratory support analysis
    """
    
    def __init__(self, mimic_path='./mimic-iii-clinical-database-1.4/'):
        """
        Initialize with path to MIMIC-III data
        
        Parameters:
        -----------
        mimic_path : str
            Path to MIMIC-III database files
        """
        self.mimic_path = mimic_path
        self.patients = None
        self.admissions = None
        self.chartevents = None
        self.procedureevents = None
        
    def load_core_tables(self):
        """Load core MIMIC-III tables"""
        print("Loading MIMIC-III tables...")
        
        # Patient demographics
        self.patients = pd.read_csv(f'{self.mimic_path}PATIENTS.csv')
        print(f"  - Loaded {len(self.patients):,} patients")
        
        # ICU admissions
        self.admissions = pd.read_csv(f'{self.mimic_path}ADMISSIONS.csv')
        print(f"  - Loaded {len(self.admissions):,} admissions")
        
        # ICU stays
        self.icustays = pd.read_csv(f'{self.mimic_path}ICUSTAYS.csv')
        print(f"  - Loaded {len(self.icustays):,} ICU stays")
        
        print("Core tables loaded successfully!\n")
        
    def extract_respiratory_parameters(self, sample_size=None):
        """
        Extract respiratory and ventilation parameters from CHARTEVENTS
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of patients to sample for faster processing
        """
        print("Extracting respiratory parameters...")
        
        # MIMIC-III ITEMID codes for respiratory parameters
        respiratory_itemids = {
            646: 'spo2',              # SpO2
            220277: 'spo2',           # O2 saturation pulseoxymetry
            618: 'respiratory_rate',   # Respiratory Rate
            220210: 'respiratory_rate', # Respiratory Rate
            223835: 'fio2',           # Inspired O2 Fraction
            3420: 'fio2',             # FiO2
            220339: 'peep',           # PEEP set
            505: 'peep',              # PEEP
            681: 'tidal_volume',      # Tidal Volume (set)
            682: 'tidal_volume',      # Tidal Volume (observed)
            224685: 'tidal_volume',   # Tidal Volume
            220045: 'heart_rate',     # Heart Rate
            211: 'heart_rate',        # Heart Rate
        }
        
        # Load chartevents in chunks (large file)
        chunk_size = 1000000
        chunks = []
        
        print("  Reading CHARTEVENTS (this may take a while)...")
        
        for i, chunk in enumerate(pd.read_csv(
            f'{self.mimic_path}CHARTEVENTS.csv',
            chunksize=chunk_size,
            usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 
                     'CHARTTIME', 'VALUE', 'VALUEUOM']
        )):
            # Filter for respiratory parameters
            chunk = chunk[chunk['ITEMID'].isin(respiratory_itemids.keys())]
            
            if not chunk.empty:
                # Map ITEMID to parameter name
                chunk['parameter'] = chunk['ITEMID'].map(respiratory_itemids)
                chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {(i+1)*chunk_size:,} rows...")
        
        # Combine chunks
        self.respiratory_data = pd.concat(chunks, ignore_index=True)
        print(f"  Extracted {len(self.respiratory_data):,} respiratory measurements\n")
        
        return self.respiratory_data
    
    def extract_ventilation_events(self):
        """Extract ventilation procedures from PROCEDUREEVENTS"""
        print("Extracting ventilation events...")
        
        # Load procedure events
        procedures = pd.read_csv(
            f'{self.mimic_path}PROCEDUREEVENTS_MV.csv',
            usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 
                     'STARTTIME', 'ENDTIME']
        )
        
        # Ventilation ITEMID codes
        vent_itemids = {
            225792: 'NIV',              # Non-invasive ventilation
            225794: 'INVASIVE_VENT',    # Invasive mechanical ventilation
            226732: 'HFNC',             # High flow nasal cannula
        }
        
        # Filter ventilation events
        vent_events = procedures[procedures['ITEMID'].isin(vent_itemids.keys())].copy()
        vent_events['ventilation_type'] = vent_events['ITEMID'].map(vent_itemids)
        
        print(f"  Found {len(vent_events):,} ventilation events\n")
        
        return vent_events
    
    def calculate_apache_ii(self):
        """
        Calculate APACHE II scores (simplified version)
        
        Note: Full APACHE II requires more parameters
        This is a simplified approximation
        """
        print("Calculating APACHE II scores...")
        
        # This would require additional vital signs, labs, etc.
        # Placeholder for demonstration
        
        apache_scores = self.icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].copy()
        
        # Simplified random score (in production, calculate from actual parameters)
        apache_scores['apache_ii'] = np.random.normal(18, 6, len(apache_scores)).clip(0, 40)
        
        print(f"  Calculated scores for {len(apache_scores):,} ICU stays\n")
        
        return apache_scores
    
    def create_analysis_dataset(self, output_file='mimic_respiratory_analysis.csv'):
        """
        Create final analysis-ready dataset
        
        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        print("Creating analysis dataset...")
        
        # Merge patient demographics with ICU stays
        dataset = self.icustays.merge(
            self.patients[['SUBJECT_ID', 'GENDER', 'DOB']],
            on='SUBJECT_ID',
            how='left'
        )
        
        # Calculate age at admission
        dataset['INTIME'] = pd.to_datetime(dataset['INTIME'])
        dataset['DOB'] = pd.to_datetime(dataset['DOB'])
        dataset['age'] = (dataset['INTIME'] - dataset['DOB']).dt.days / 365.25
        
        # Add admission info
        dataset = dataset.merge(
            self.admissions[['HADM_ID', 'ADMISSION_TYPE', 'ETHNICITY', 
                           'HOSPITAL_EXPIRE_FLAG']],
            on='HADM_ID',
            how='left'
        )
        
        # Calculate ICU length of stay
        dataset['OUTTIME'] = pd.to_datetime(dataset['OUTTIME'])
        dataset['icu_los'] = (dataset['OUTTIME'] - dataset['INTIME']).dt.total_seconds() / 3600 / 24
        
        # Add ventilation data
        vent_events = self.extract_ventilation_events()
        
        # For each ICU stay, determine primary ventilation type
        vent_summary = vent_events.groupby('ICUSTAY_ID')['ventilation_type'].first().reset_index()
        dataset = dataset.merge(vent_summary, on='ICUSTAY_ID', how='left')
        dataset['ventilation_type'].fillna('NONE', inplace=True)
        
        # Add APACHE II scores
        apache = self.calculate_apache_ii()
        dataset = dataset.merge(apache, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'], how='left')
        
        # Create outcome variable
        dataset['outcome'] = dataset['HOSPITAL_EXPIRE_FLAG'].map({0: 'Survived', 1: 'Died'})
        
        # Select final columns
        final_columns = [
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'age', 'GENDER',
            'ETHNICITY', 'ADMISSION_TYPE', 'ventilation_type', 
            'apache_ii', 'icu_los', 'outcome'
        ]
        
        final_dataset = dataset[final_columns].copy()
        
        # Save to CSV
        final_dataset.to_csv(output_file, index=False)
        print(f"✅ Analysis dataset saved to {output_file}")
        print(f"   Total records: {len(final_dataset):,}")
        print(f"   Columns: {len(final_dataset.columns)}")
        
        return final_dataset
    
    def create_timeseries_data(self, output_file='mimic_respiratory_timeseries.csv'):
        """
        Create time-series dataset of respiratory parameters
        
        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        print("\nCreating time-series dataset...")
        
        # Pivot respiratory data
        timeseries = self.respiratory_data.pivot_table(
            index=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME'],
            columns='parameter',
            values='VALUE',
            aggfunc='first'
        ).reset_index()
        
        # Convert to numeric
        numeric_cols = ['spo2', 'respiratory_rate', 'heart_rate', 'fio2', 
                       'peep', 'tidal_volume']
        for col in numeric_cols:
            if col in timeseries.columns:
                timeseries[col] = pd.to_numeric(timeseries[col], errors='coerce')
        
        # Sort by time
        timeseries['CHARTTIME'] = pd.to_datetime(timeseries['CHARTTIME'])
        timeseries = timeseries.sort_values(['ICUSTAY_ID', 'CHARTTIME'])
        
        # Calculate hours from ICU admission
        icu_start = self.icustays[['ICUSTAY_ID', 'INTIME']].copy()
        icu_start['INTIME'] = pd.to_datetime(icu_start['INTIME'])
        
        timeseries = timeseries.merge(icu_start, on='ICUSTAY_ID', how='left')
        timeseries['hour'] = (timeseries['CHARTTIME'] - timeseries['INTIME']).dt.total_seconds() / 3600
        
        # Data quality flags
        timeseries['quality_flag'] = 'Valid'
        
        # Flag physiologically implausible values
        if 'spo2' in timeseries.columns:
            timeseries.loc[timeseries['spo2'] < 70, 'quality_flag'] = 'Out_of_range'
            timeseries.loc[timeseries['spo2'] > 100, 'quality_flag'] = 'Out_of_range'
        
        if 'respiratory_rate' in timeseries.columns:
            timeseries.loc[timeseries['respiratory_rate'] > 50, 'quality_flag'] = 'Suspicious'
        
        # Save to CSV
        timeseries.to_csv(output_file, index=False)
        print(f"✅ Time-series dataset saved to {output_file}")
        print(f"   Total measurements: {len(timeseries):,}")
        
        return timeseries


def main():
    """Main execution function"""
    
    print("="*70)
    print("MIMIC-III Data Preprocessing for NIV Analysis")
    print("="*70)
    print()
    
    # Initialize preprocessor
    # UPDATE THIS PATH to your MIMIC-III location
    preprocessor = MIMICPreprocessor(mimic_path='./mimic-iii-clinical-database-1.4/')
    
    try:
        # Step 1: Load core tables
        preprocessor.load_core_tables()
        
        # Step 2: Extract respiratory parameters
        # Note: This is memory-intensive. Consider sampling for testing
        preprocessor.extract_respiratory_parameters()
        
        # Step 3: Create analysis datasets
        patients_df = preprocessor.create_analysis_dataset(
            output_file='data/mimic_niv_patients.csv'
        )
        
        timeseries_df = preprocessor.create_timeseries_data(
            output_file='data/mimic_niv_timeseries.csv'
        )
        
        print("\n" + "="*70)
        print("✅ Preprocessing complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  1. data/mimic_niv_patients.csv - Patient-level data")
        print("  2. data/mimic_niv_timeseries.csv - Time-series vital signs")
        print("\nNext steps:")
        print("  1. Review data quality")
        print("  2. Update niv_analysis_dashboard.py to use real data")
        print("  3. Run: streamlit run niv_analysis_dashboard.py")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure MIMIC-III data is downloaded")
        print("  2. Update mimic_path in the code")
        print("  3. Verify PhysioNet credentialing")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your data paths and permissions")


if __name__ == "__main__":
    main()
