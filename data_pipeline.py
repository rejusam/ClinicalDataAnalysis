"""
Automated Data Pipeline
Orchestrates the full ETL workflow with validation gates and logging.

Stages:
  1. Source validation   - verify raw data exists
  2. ETL processing      - extract, transform, load vital signs
  3. Quality validation  - completeness, range, and duplicate checks
  4. Power BI export     - generate star-schema datasets

Usage:
  python data_pipeline.py                 # full pipeline (requires raw data)
  python data_pipeline.py --skip-etl      # skip ETL, use existing CSVs
  python data_pipeline.py --export-only   # regenerate Power BI exports only

Author: Reju Sam John
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR, RAW_DATA_DIR, VITALS_FILE, SUMMARY_FILE,
    POWERBI_EXPORT_DIR, MAX_HOURS, SAMPLE_SIZE,
    SPO2_MIN_VALID, SPO2_MAX_VALID, COMPLETENESS_TARGET,
)
from mimic_waveform_processor import MIMICWaveformProcessor
from powerbi_export import PowerBIExporter

# --- Logging ---
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """End-to-end pipeline with validation gates between stages."""

    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.status = {"run_id": self.run_id, "steps": {}}

    def _log_step(self, step, result, details=""):
        self.status["steps"][step] = {
            "status": result,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        level = logging.ERROR if result == "FAIL" else logging.INFO
        logger.log(level, "[%s] %s - %s", step, result, details)

    # ---- Gate 1 ----
    def validate_source(self):
        logger.info("=" * 60)
        logger.info("GATE 1: Source Data Validation")

        if not RAW_DATA_DIR.exists():
            self._log_step("source_validation", "FAIL",
                           f"Directory not found: {RAW_DATA_DIR}")
            return False

        patients = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
        if not patients:
            self._log_step("source_validation", "FAIL",
                           "No patient directories found")
            return False

        self._log_step("source_validation", "PASS",
                       f"{len(patients)} patient directories")
        return True

    # ---- Gate 2 ----
    def run_etl(self):
        logger.info("=" * 60)
        logger.info("GATE 2: ETL Processing")

        processor = MIMICWaveformProcessor(data_path=str(RAW_DATA_DIR))
        vitals = processor.process_all_patients(
            sample_size=SAMPLE_SIZE, max_hours=MAX_HOURS,
        )

        if vitals is None or len(vitals) == 0:
            self._log_step("etl", "FAIL", "No data extracted")
            return None, None

        summary = processor.create_patient_summary(vitals)
        processor.save_processed_data(
            vitals, summary,
            vitals_file=str(VITALS_FILE),
            summary_file=str(SUMMARY_FILE),
        )
        self._log_step(
            "etl", "PASS",
            f"{len(vitals):,} measurements, {len(summary)} patients",
        )
        return vitals, summary

    # ---- Gate 3 ----
    def validate_quality(self, vitals):
        logger.info("=" * 60)
        logger.info("GATE 3: Output Quality Validation")

        issues = []
        for col in ['spo2', 'respiratory_rate', 'heart_rate']:
            if col not in vitals.columns:
                continue
            pct = vitals[col].notna().mean() * 100
            if pct < COMPLETENESS_TARGET:
                issues.append(f"{col} completeness {pct:.1f}% < target")

        if 'spo2' in vitals.columns:
            spo2 = vitals['spo2'].dropna()
            oor = ((spo2 < SPO2_MIN_VALID) | (spo2 > SPO2_MAX_VALID)).mean() * 100
            if oor > 5:
                issues.append(f"SpO2 out-of-range {oor:.1f}%")

        dup_cols = ['subject_id', 'hours', 'minutes']
        available = [c for c in dup_cols if c in vitals.columns]
        if available:
            dups = vitals.duplicated(subset=available, keep=False).sum()
            if dups:
                issues.append(f"{dups} duplicate records")

        result = "WARN" if issues else "PASS"
        self._log_step("quality", result,
                       "; ".join(issues) if issues else "All checks passed")
        return not issues

    # ---- Gate 4 ----
    def export_powerbi(self):
        logger.info("=" * 60)
        logger.info("GATE 4: Power BI Export")

        try:
            exporter = PowerBIExporter(
                vitals_path=str(VITALS_FILE),
                summary_path=str(SUMMARY_FILE),
                output_dir=str(POWERBI_EXPORT_DIR),
            )
            exporter.export_all()
            self._log_step("powerbi_export", "PASS",
                           f"Files -> {POWERBI_EXPORT_DIR}")
            return True
        except Exception as e:
            self._log_step("powerbi_export", "FAIL", str(e))
            return False

    # ---- Report ----
    def report(self):
        self.status["completed"] = datetime.now().isoformat()
        self.status["overall"] = (
            "SUCCESS"
            if all(s["status"] in ("PASS", "WARN", "SKIP")
                   for s in self.status["steps"].values())
            else "FAILED"
        )
        path = LOG_DIR / f"pipeline_report_{self.run_id}.json"
        with open(path, 'w') as f:
            json.dump(self.status, f, indent=2)

        logger.info("Pipeline report: %s", path)
        logger.info("Overall: %s", self.status["overall"])
        return self.status

    # ---- Entrypoint ----
    def run(self, skip_etl=False):
        logger.info("=" * 60)
        logger.info("DATA PIPELINE  -  Run %s", self.run_id)
        logger.info("=" * 60)

        if not skip_etl:
            if not self.validate_source():
                logger.info("Raw data unavailable; falling back to processed CSVs")
                skip_etl = True

        if not skip_etl:
            vitals, _ = self.run_etl()
            if vitals is None:
                return self.report()
            self.validate_quality(vitals)
        else:
            self._log_step("etl", "SKIP", "Using existing processed data")
            if VITALS_FILE.exists():
                self.validate_quality(pd.read_csv(str(VITALS_FILE)))
            else:
                self._log_step("quality", "FAIL", "No processed data found")
                return self.report()

        self.export_powerbi()
        return self.report()


def main():
    parser = argparse.ArgumentParser(description="Clinical Data Pipeline")
    parser.add_argument('--skip-etl', action='store_true',
                        help='Use existing processed CSVs instead of re-running ETL')
    parser.add_argument('--export-only', action='store_true',
                        help='Only regenerate Power BI exports')
    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.export_only:
        pipeline.export_powerbi()
        pipeline.report()
    else:
        pipeline.run(skip_etl=args.skip_etl)


if __name__ == "__main__":
    main()
