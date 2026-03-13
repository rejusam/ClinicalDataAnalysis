"""
Unit tests for the data processing pipeline.

Author: Reju Sam John
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mimic_waveform_processor import MIMICWaveformProcessor
from config import (
    SPO2_HYPOXEMIA_THRESHOLD,
    RR_TACHYPNEA_THRESHOLD,
    SPO2_MIN_VALID,
    SPO2_MAX_VALID,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def sample_vitals():
    """Synthetic minute-level vital signs for one patient."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'subject_id': ['p000001'] * n,
        'record_name': ['p000001n'] * n,
        'hours': np.arange(n) / 60,
        'minutes': np.arange(n),
        'HR': np.random.normal(80, 10, n),
        'SpO2': np.random.normal(95, 3, n).clip(70, 100),
        'RESP': np.random.normal(18, 4, n).clip(5, 40),
    })


@pytest.fixture
def processor():
    return MIMICWaveformProcessor(data_path='./data/test')


# ---------------------------------------------------------------
# Column standardization
# ---------------------------------------------------------------

class TestVitalSignStandardization:

    def test_hr_mapped(self, processor, sample_vitals):
        result = processor.standardize_vital_signs(sample_vitals.copy())
        assert 'heart_rate' in result.columns
        assert 'HR' not in result.columns

    def test_spo2_mapped(self, processor, sample_vitals):
        result = processor.standardize_vital_signs(sample_vitals.copy())
        assert 'spo2' in result.columns
        assert 'SpO2' not in result.columns

    def test_rr_mapped(self, processor, sample_vitals):
        result = processor.standardize_vital_signs(sample_vitals.copy())
        assert 'respiratory_rate' in result.columns
        assert 'RESP' not in result.columns

    def test_duplicate_sources_merged(self, processor):
        df = pd.DataFrame({
            'HR': [80, 85, np.nan],
            'PULSE': [np.nan, np.nan, 92],
            'SpO2': [95, 96, 97],
        })
        result = processor.standardize_vital_signs(df)
        assert 'heart_rate' in result.columns
        assert 'PULSE' not in result.columns
        assert result['heart_rate'].iloc[2] == 92  # filled from PULSE


# ---------------------------------------------------------------
# Derived respiratory metrics
# ---------------------------------------------------------------

class TestRespiratoryMetrics:

    def test_hypoxemia_flag(self, processor, sample_vitals):
        df = processor.standardize_vital_signs(sample_vitals.copy())
        result = processor.calculate_respiratory_metrics(df)

        expected = (result['spo2'] < SPO2_HYPOXEMIA_THRESHOLD).astype(int)
        pd.testing.assert_series_equal(
            result['hypoxemia'], expected, check_names=False
        )

    def test_tachypnea_flag(self, processor, sample_vitals):
        df = processor.standardize_vital_signs(sample_vitals.copy())
        result = processor.calculate_respiratory_metrics(df)

        expected = (result['respiratory_rate'] > RR_TACHYPNEA_THRESHOLD).astype(int)
        pd.testing.assert_series_equal(
            result['tachypnea'], expected, check_names=False
        )

    def test_spo2_variability_present(self, processor, sample_vitals):
        df = processor.standardize_vital_signs(sample_vitals.copy())
        result = processor.calculate_respiratory_metrics(df)
        assert 'spo2_variability' in result.columns
        assert result['spo2_variability'].notna().any()

    def test_no_hypoxemia_when_normal(self, processor):
        df = pd.DataFrame({
            'spo2': [95, 96, 97, 98, 99],
            'respiratory_rate': [18, 19, 20, 18, 17],
        })
        result = processor.calculate_respiratory_metrics(df)
        assert result['hypoxemia'].sum() == 0


# ---------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------

class TestDataQuality:

    def test_spo2_range_validation(self):
        values = pd.Series([50, 70, 85, 95, 100, 105])
        valid = (values >= SPO2_MIN_VALID) & (values <= SPO2_MAX_VALID)
        assert valid.sum() == 4  # 70, 85, 95, 100

    def test_no_null_subject_ids(self, sample_vitals):
        assert sample_vitals['subject_id'].notna().all()

    def test_hours_monotonic(self, sample_vitals):
        assert sample_vitals['hours'].is_monotonic_increasing


# ---------------------------------------------------------------
# Patient summary
# ---------------------------------------------------------------

class TestPatientSummary:

    def test_summary_has_expected_columns(self, processor, sample_vitals):
        df = processor.standardize_vital_signs(sample_vitals.copy())
        df = processor.calculate_respiratory_metrics(df)
        df['ventilation_type'] = 'NIV'

        summary = processor.create_patient_summary(df)
        assert len(summary) == 1
        assert 'spo2_mean' in summary.columns
        assert 'hypoxemia_rate' in summary.columns

    def test_summary_spo2_in_range(self, processor, sample_vitals):
        df = processor.standardize_vital_signs(sample_vitals.copy())
        df = processor.calculate_respiratory_metrics(df)
        df['ventilation_type'] = 'NIV'

        summary = processor.create_patient_summary(df)
        assert 70 <= summary['spo2_mean'].iloc[0] <= 100
