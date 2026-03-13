# Respiratory Biometrics Analytics Platform

An automated data analytics pipeline that processes 34,000+ ICU vital-sign measurements, validates data quality, and delivers insights through an interactive Streamlit dashboard and Power BI-ready star-schema exports.

Built to demonstrate end-to-end analytics delivery: from raw data ingestion and automated ETL, through statistical analysis, to self-service BI-ready outputs with CI/CD lifecycle management.

---

## Architecture

```
Raw Data (PhysioNet MIMIC-III)
        |
        v
  +-----------------+       +------------------+
  | get_data_data.sh|  -->  | mimic_waveform_  |
  | (data download) |       | processor.py     |
  +-----------------+       | (ETL + feature   |
                            |  engineering)    |
                            +--------+---------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
            +------------------+         +-------------------+
            | niv_streamlit_   |         | powerbi_export.py |
            | app.py           |         | (star schema +    |
            | (interactive     |         |  DAX measures)    |
            |  dashboard)      |         +-------------------+
            +------------------+                  |
                                                  v
                                         Power BI Desktop
                                         (dim/fact CSVs)
```

Orchestrated by **`data_pipeline.py`** with validation gates, logging, and CI/CD via GitHub Actions.

---

## Key Features

| Capability                  | Implementation                                                   |
|-----------------------------|------------------------------------------------------------------|
| **Automated ETL**           | Python pipeline with validation gates and structured logging     |
| **Interactive Dashboard**   | 5-module Streamlit app with Plotly visualisations                 |
| **Power BI Integration**    | Star-schema export (fact + dimension tables) with DAX templates  |
| **Statistical Analysis**    | t-tests, Cohen's d, correlation matrices, Z-score outlier detection |
| **Data Quality**            | Completeness checks, range validation, quality flags             |
| **CI/CD**                   | GitHub Actions: lint, test, data validation on every push        |
| **Configuration Management**| Centralised `config.py` for thresholds, paths, and parameters    |
| **Testing**                 | pytest suite covering standardisation, metrics, and summaries    |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the automated pipeline (uses existing processed data)
python data_pipeline.py --skip-etl

# Launch the interactive dashboard
streamlit run niv_streamlit_app.py

# Generate Power BI exports
python powerbi_export.py

# Run tests
pytest tests/ -v
```

Or use the Makefile shortcuts:

```bash
make install    # pip install
make pipeline   # run pipeline with existing data
make run        # launch Streamlit dashboard
make export     # generate Power BI datasets
make test       # run pytest suite
```

---

## Data Pipeline

The pipeline (`data_pipeline.py`) runs through four validation gates:

```
Gate 1: Source Validation      - Verify raw data directories exist
Gate 2: ETL Processing         - Extract, standardise, engineer features
Gate 3: Quality Validation     - Completeness, range checks, duplicates
Gate 4: Power BI Export        - Generate star-schema CSVs + DAX measures
```

Each gate logs its outcome (PASS / WARN / FAIL) with timestamps to `data/logs/`. A JSON report is generated per run for audit and debugging.

### Pipeline Modes

```bash
python data_pipeline.py                # Full pipeline (requires raw MIMIC data)
python data_pipeline.py --skip-etl     # Skip ETL, validate + export existing data
python data_pipeline.py --export-only  # Regenerate Power BI exports only
```

---

## ETL Processing

**Script:** `mimic_waveform_processor.py`

Processes raw MIMIC-III waveform records into analysis-ready datasets:

1. **Extraction** - Reads minute-by-minute vital signs via the WFDB library
2. **Standardisation** - Maps inconsistent monitor nomenclature to standard names:
   - `HR`, `PULSE` -> `heart_rate`
   - `SpO2`, `SAO2` -> `spo2`
   - `RESP`, `RR` -> `respiratory_rate`
3. **Feature Engineering** - Derives clinical indicators:
   - Hypoxemia detection (SpO2 < 90%)
   - Tachypnea / bradypnea flags
   - SpO2 variability (10-minute rolling std)
   - Ventilation type classification
4. **Quality Flags** - Range validation and Z-score outlier detection

### Output Datasets

| File                          | Rows    | Description                               |
|-------------------------------|---------|-------------------------------------------|
| `mimic_waveform_vitals.csv`   | 34,630  | Minute-level time-series measurements     |
| `mimic_waveform_summary.csv`  | 10      | Patient-level aggregated statistics       |

---

## Power BI Integration

**Script:** `powerbi_export.py`

Generates a star-schema data model optimised for Power BI:

```
                  dim_patient
                      |
fact_vitals ---+------+------+--- dim_time
               |
           dim_quality
```

### Exported Files

| File                    | Description                                    |
|-------------------------|------------------------------------------------|
| `dim_patient.csv`       | Patient dimension with risk stratification     |
| `dim_time.csv`          | Hour/shift/day/period dimension                |
| `dim_quality.csv`       | Quality flag dimension for slicer filtering    |
| `fact_vitals.csv`       | Fact table with surrogate keys                 |
| `agg_hourly_vitals.csv` | Pre-aggregated hourly rollups for performance  |
| `dax_measures.json`     | Ready-to-paste DAX measure definitions         |
| `data_model.json`       | Relationships and model documentation          |

### Sample DAX Measures

```dax
Average SpO2 = AVERAGE(fact_vitals[spo2])

Hypoxemia Rate % =
    DIVIDE(
        COUNTROWS(FILTER(fact_vitals, fact_vitals[hypoxemia] = 1)),
        COUNTROWS(fact_vitals)
    ) * 100

High Risk Patient Count =
    CALCULATE(
        DISTINCTCOUNT(dim_patient[patient_key]),
        dim_patient[risk_category] = "High"
    )
```

---

## Interactive Dashboard

**Script:** `niv_streamlit_app.py` | Launch: `streamlit run niv_streamlit_app.py`

Five integrated analysis modules:

| Module                       | What it delivers                                          |
|------------------------------|-----------------------------------------------------------|
| **Overview & Demographics**  | Cohort KPIs, severity gauges, monitoring duration stats    |
| **Respiratory Biometrics**   | SpO2 and RR distributions, patient-level comparisons      |
| **Temporal Analysis**        | Per-patient time-series trajectories, hypoxemia episodes   |
| **Data Quality Report**      | Completeness rates, quality flags, outlier detection       |
| **Statistical Analysis**     | t-tests, effect sizes, correlation heatmap, outcome analysis |

Features: interactive Plotly charts, CSV download, real-time patient filtering, colour-coded clinical thresholds.

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Lint** - flake8 syntax and style checks
2. **Test** - pytest unit tests for the processing pipeline
3. **Data Validation** - Schema and range checks on the committed CSV datasets

---

## Project Structure

```
ClinicalDataAnalysis/
|-- .github/workflows/ci.yml      # CI/CD pipeline
|-- .devcontainer/devcontainer.json # Codespaces / Dev Container setup
|-- data/
|   |-- mimic_waveform_vitals.csv  # Processed time-series (34,630 rows)
|   |-- mimic_waveform_summary.csv # Patient summaries (10 rows)
|   +-- powerbi_exports/           # Generated star-schema files
|-- tests/
|   +-- test_processor.py          # Unit tests
|-- config.py                      # Centralised configuration
|-- data_pipeline.py               # Automated pipeline orchestrator
|-- get_data_data.sh               # Raw data download script
|-- mimic_waveform_processor.py    # ETL and feature engineering
|-- niv_streamlit_app.py           # Interactive Streamlit dashboard
|-- powerbi_export.py              # Power BI data model export
|-- requirements.txt               # Python dependencies
|-- Makefile                        # Common task shortcuts
+-- README.md
```

---

## Tech Stack

| Layer            | Technology                                     |
|------------------|------------------------------------------------|
| Language         | Python 3.11                                    |
| Data Processing  | Pandas, NumPy                                  |
| Visualisation    | Plotly, Streamlit                               |
| Statistics       | SciPy                                          |
| Waveform I/O     | WFDB                                           |
| BI Export         | Star-schema CSVs for Power BI                  |
| CI/CD            | GitHub Actions                                 |
| Testing          | pytest, flake8                                 |
| Dev Environment  | VS Code Dev Containers                         |

---

## Data Source

**MIMIC-III Waveform Database Matched Subset** (PhysioNet)

10 ICU patient records with continuous minute-by-minute vital signs:
- SpO2 (oxygen saturation), respiratory rate, heart rate
- Arterial and non-invasive blood pressure
- Ventilation parameters

To download raw data (requires PhysioNet credentials):

```bash
bash get_data_data.sh
```

### References

1. Moody, B. et al. (2020). MIMIC-III Waveform Database Matched Subset. PhysioNet. DOI: 10.13026/C2294B
2. Johnson, A.E.W. et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.

---

## Author

**Reju Sam John**
rejusamjohn@gmail.com | [LinkedIn](https://www.linkedin.com/in/dr-reju-sam-john-14b00774/)
