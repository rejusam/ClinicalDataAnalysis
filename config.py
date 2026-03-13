"""
Configuration for the Clinical Data Analysis Pipeline.
Centralizes data paths, clinical thresholds, and export parameters.

Author: Reju Sam John
"""

from pathlib import Path

# --- Data Paths ---
DATA_DIR = Path("./data")
RAW_DATA_DIR = DATA_DIR / "mimic3wdb" / "p00"
VITALS_FILE = DATA_DIR / "mimic_waveform_vitals.csv"
SUMMARY_FILE = DATA_DIR / "mimic_waveform_summary.csv"
POWERBI_EXPORT_DIR = DATA_DIR / "powerbi_exports"

# --- Processing Parameters ---
MAX_HOURS = 72
SAMPLE_SIZE = 50

# --- Clinical Thresholds ---
SPO2_HYPOXEMIA_THRESHOLD = 90
SPO2_TARGET = 92
RR_TACHYPNEA_THRESHOLD = 24
RR_BRADYPNEA_THRESHOLD = 12

# --- Valid Ranges (physiologically plausible) ---
HR_MIN_VALID = 30
HR_MAX_VALID = 200
SPO2_MIN_VALID = 70
SPO2_MAX_VALID = 100
RR_MIN_VALID = 5
RR_MAX_VALID = 50

# --- Quality Parameters ---
OUTLIER_ZSCORE_THRESHOLD = 3
SPO2_VARIABILITY_WINDOW = 10
COMPLETENESS_TARGET = 95.0

# --- Export Settings ---
POWERBI_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CSV_ENCODING = "utf-8"
