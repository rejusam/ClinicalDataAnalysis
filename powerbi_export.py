"""
Power BI Data Export Module
Prepares analysis-ready datasets in a star schema for Power BI dashboards.

Creates:
- Fact table:      minute-level vital sign measurements
- Dimension tables: patients, time periods, quality flags
- Pre-aggregated hourly measures for report performance
- DAX measure templates and data model documentation

Author: Reju Sam John
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PowerBIExporter:
    """Transforms processed clinical data into Power BI-optimized datasets."""

    def __init__(self, vitals_path, summary_path, output_dir='./data/powerbi_exports'):
        self.vitals_df = pd.read_csv(vitals_path)
        self.summary_df = pd.read_csv(summary_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dimension tables
    # ------------------------------------------------------------------

    def create_dim_patient(self):
        """Patient dimension with risk stratification."""
        logger.info("Building dim_patient...")

        dim = self.summary_df[['subject_id']].copy()
        dim['patient_key'] = range(1, len(dim) + 1)

        if 'ventilation_type' in self.summary_df.columns:
            dim['ventilation_type'] = self.summary_df['ventilation_type']
        dim['monitoring_hours'] = self.summary_df['monitoring_hours']

        dim['risk_category'] = pd.cut(
            self.summary_df['hypoxemia_rate'],
            bins=[-1, 10, 30, 100],
            labels=['Low', 'Moderate', 'High']
        )
        dim['spo2_mean'] = self.summary_df['spo2_mean']
        dim['hypoxemia_rate'] = self.summary_df['hypoxemia_rate']

        return dim

    def create_dim_time(self):
        """Time dimension for hourly slicing and shift analysis."""
        logger.info("Building dim_time...")

        max_hours = int(self.vitals_df['hours'].max()) + 1
        hours = list(range(max_hours))

        dim = pd.DataFrame({
            'hour_key': hours,
            'hour_of_stay': hours,
            'shift': [
                'Night' if h % 24 < 7 else ('Day' if h % 24 < 19 else 'Night')
                for h in hours
            ],
            'day_of_stay': [h // 24 + 1 for h in hours],
            'period': [
                'First 24h' if h < 24
                else '24-48h' if h < 48
                else '48-72h'
                for h in hours
            ]
        })
        return dim

    def create_dim_quality(self):
        """Quality-flag dimension for slicer filtering."""
        logger.info("Building dim_quality...")
        flags = ['Valid', 'Out_of_range', 'Suspicious']
        return pd.DataFrame({
            'quality_key': range(1, len(flags) + 1),
            'quality_flag': flags,
            'is_valid_for_analysis': [True, False, False]
        })

    # ------------------------------------------------------------------
    # Fact table
    # ------------------------------------------------------------------

    def create_fact_vitals(self, dim_patient):
        """Fact table with foreign keys to each dimension."""
        logger.info("Building fact_vitals...")

        patient_map = dict(zip(dim_patient['subject_id'], dim_patient['patient_key']))
        quality_map = {'Valid': 1, 'Out_of_range': 2, 'Suspicious': 3}

        fact = self.vitals_df.copy()
        fact['patient_key'] = fact['subject_id'].map(patient_map)
        fact['hour_key'] = fact['hours'].round(0).astype(int)

        if 'quality_flag' not in fact.columns:
            fact['quality_flag'] = 'Valid'
            if 'spo2' in fact.columns:
                fact.loc[fact['spo2'] < 70, 'quality_flag'] = 'Out_of_range'
                fact.loc[fact['spo2'] > 100, 'quality_flag'] = 'Out_of_range'

        fact['quality_key'] = fact['quality_flag'].map(quality_map).fillna(1).astype(int)

        keep = [
            'patient_key', 'hour_key', 'quality_key',
            'heart_rate', 'respiratory_rate', 'spo2',
            'hypoxemia', 'tachypnea', 'spo2_variability'
        ]
        return fact[[c for c in keep if c in fact.columns]]

    # ------------------------------------------------------------------
    # Pre-aggregated measures (improves Power BI report speed)
    # ------------------------------------------------------------------

    def create_hourly_aggregates(self):
        """Hourly rollups for trend visuals without hitting row-level data."""
        logger.info("Computing hourly aggregates...")

        hourly = self.vitals_df.groupby([
            'subject_id',
            self.vitals_df['hours'].round(0).astype(int)
        ]).agg({
            'spo2': ['mean', 'std', 'min', 'max'],
            'respiratory_rate': ['mean', 'max'],
            'heart_rate': ['mean', 'max'],
            'hypoxemia': 'sum'
        }).reset_index()

        hourly.columns = [
            'subject_id', 'hour',
            'spo2_mean', 'spo2_std', 'spo2_min', 'spo2_max',
            'rr_mean', 'rr_max',
            'hr_mean', 'hr_max',
            'hypoxemia_minutes'
        ]
        return hourly

    # ------------------------------------------------------------------
    # DAX + documentation
    # ------------------------------------------------------------------

    def generate_dax_measures(self):
        """Starter DAX measures ready to paste into Power BI Desktop."""
        return {
            "Average SpO2": "AVERAGE(fact_vitals[spo2])",
            "Hypoxemia Rate %": (
                "DIVIDE("
                "COUNTROWS(FILTER(fact_vitals, fact_vitals[hypoxemia] = 1)), "
                "COUNTROWS(fact_vitals)) * 100"
            ),
            "Patient Count": "DISTINCTCOUNT(fact_vitals[patient_key])",
            "Total Monitoring Hours": "SUM(dim_patient[monitoring_hours])",
            "SpO2 Below Target": (
                "CALCULATE(COUNTROWS(fact_vitals), fact_vitals[spo2] < 92)"
            ),
            "Tachypnea Rate %": (
                "DIVIDE("
                "COUNTROWS(FILTER(fact_vitals, fact_vitals[tachypnea] = 1)), "
                "COUNTROWS(fact_vitals)) * 100"
            ),
            "Mean Heart Rate": "AVERAGE(fact_vitals[heart_rate])",
            "High Risk Patient Count": (
                "CALCULATE("
                "DISTINCTCOUNT(dim_patient[patient_key]), "
                "dim_patient[risk_category] = \"High\")"
            ),
        }

    def generate_data_model_doc(self, dim_patient, dim_time, dim_quality, fact):
        """Machine-readable data model documentation."""
        return {
            "model_name": "ICU Respiratory Biometrics",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "author": "Reju Sam John",
            "tables": {
                "fact_vitals": {
                    "type": "Fact",
                    "rows": len(fact),
                    "grain": "One row per patient per minute",
                    "relationships": [
                        {"to": "dim_patient", "key": "patient_key",
                         "cardinality": "many-to-one"},
                        {"to": "dim_time", "key": "hour_key",
                         "cardinality": "many-to-one"},
                        {"to": "dim_quality", "key": "quality_key",
                         "cardinality": "many-to-one"},
                    ]
                },
                "dim_patient": {
                    "type": "Dimension", "rows": len(dim_patient),
                    "grain": "One row per patient"
                },
                "dim_time": {
                    "type": "Dimension", "rows": len(dim_time),
                    "grain": "One row per hour of ICU stay"
                },
                "dim_quality": {
                    "type": "Dimension", "rows": len(dim_quality),
                    "grain": "One row per quality classification"
                },
            }
        }

    # ------------------------------------------------------------------
    # Full export
    # ------------------------------------------------------------------

    def export_all(self):
        """Run the complete export pipeline and write to disk."""
        logger.info("Starting Power BI export...")

        dim_patient = self.create_dim_patient()
        dim_time = self.create_dim_time()
        dim_quality = self.create_dim_quality()
        fact = self.create_fact_vitals(dim_patient)
        hourly = self.create_hourly_aggregates()

        # Write CSVs
        dim_patient.to_csv(self.output_dir / 'dim_patient.csv', index=False)
        dim_time.to_csv(self.output_dir / 'dim_time.csv', index=False)
        dim_quality.to_csv(self.output_dir / 'dim_quality.csv', index=False)
        fact.to_csv(self.output_dir / 'fact_vitals.csv', index=False)
        hourly.to_csv(self.output_dir / 'agg_hourly_vitals.csv', index=False)

        # Write DAX measures
        dax = self.generate_dax_measures()
        with open(self.output_dir / 'dax_measures.json', 'w') as f:
            json.dump(dax, f, indent=2)

        # Write data model docs
        model = self.generate_data_model_doc(dim_patient, dim_time, dim_quality, fact)
        with open(self.output_dir / 'data_model.json', 'w') as f:
            json.dump(model, f, indent=2)

        logger.info("Export complete -> %s", self.output_dir)
        logger.info("  dim_patient.csv       %d rows", len(dim_patient))
        logger.info("  dim_time.csv          %d rows", len(dim_time))
        logger.info("  dim_quality.csv       %d rows", len(dim_quality))
        logger.info("  fact_vitals.csv       %d rows", len(fact))
        logger.info("  agg_hourly_vitals.csv %d rows", len(hourly))
        logger.info("  dax_measures.json     %d measures", len(dax))
        logger.info("  data_model.json       model documentation")

        return {
            'dim_patient': dim_patient,
            'dim_time': dim_time,
            'dim_quality': dim_quality,
            'fact_vitals': fact,
            'hourly_agg': hourly,
        }


def main():
    """Export data for Power BI consumption."""
    print("=" * 60)
    print("Power BI Data Export")
    print("=" * 60)

    exporter = PowerBIExporter(
        vitals_path='./data/mimic_waveform_vitals.csv',
        summary_path='./data/mimic_waveform_summary.csv'
    )
    exporter.export_all()

    print("\nDAX Measures (paste into Power BI Desktop):")
    print("-" * 40)
    for name, formula in exporter.generate_dax_measures().items():
        print(f"  {name} = {formula}")


if __name__ == "__main__":
    main()
