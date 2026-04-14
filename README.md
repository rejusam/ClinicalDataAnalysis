# Respiratory Biometrics Analytics Platform

**Live dashboard:** [rejusam-clinicaldataanalysis-niv-streamlit-app-mggky8.streamlit.app](https://rejusam-clinicaldataanalysis-niv-streamlit-app-mggky8.streamlit.app/)

An automated data analytics pipeline that processes 34,000+ ICU vital-sign measurements, validates data quality, and delivers insights through an interactive Streamlit dashboard and Power BI-ready star-schema exports.

Built to demonstrate end-to-end analytics delivery: from raw data ingestion and automated ETL, through statistical analysis, to self-service BI-ready outputs with CI/CD lifecycle management.

---

## Business Impact & Stakeholder View

The platform is designed around the question a Data Centre of Excellence has to answer every week: *how do we turn a messy clinical data source into something a business stakeholder can trust, explore, and act on — without a data analyst in the loop for every question?*

### What the platform delivers

| Stakeholder                 | Their question today                                       | What this platform gives them                                    |
|-----------------------------|------------------------------------------------------------|------------------------------------------------------------------|
| **Clinical lead**           | *"Which patients spent the most time below SpO2 target?"*  | Risk-stratified patient view, hypoxemia rate per patient, drill-down to minute-level readings |
| **Quality & safety team**   | *"Are our monitoring data actually reliable enough to report on?"* | Automated quality report (completeness, range, outlier flags) with PASS/WARN/FAIL status |
| **Operations / shift lead** | *"Does respiratory deterioration cluster by shift or day of stay?"* | Temporal breakdown by shift, day-of-stay, and cohort period (first 24h / 24–48h / 48–72h) |
| **BI / reporting analyst**  | *"I need this in Power BI with proper measures, not a pile of CSVs."* | Star-schema exports with surrogate keys, DAX measure templates, model documentation |
| **Data engineer on-call**   | *"Did last night's refresh actually succeed?"*             | Timestamped JSON run reports, structured logs per gate, exit codes for alerting |

### What changes when this platform is in place

- **From one-off analyses to a living data product.** The same pipeline runs on every code change (via CI) and every new data drop. Reports don't drift away from the code that produced them.
- **From "ask the analyst" to managed self-service.** The Power BI star schema is deliberately shaped so business users can build their own visuals. The DAX starter measures give them a correct baseline without them needing to understand the underlying filter context.
- **From hope-based quality to observable quality.** Four named validation gates, each with a logged outcome. If something drifts, you see it in the next run's report, not three weeks later in a stakeholder email.
- **From local notebook to reproducible build.** Centralised config, pinned requirements, containerised dev environment, CI gating merges. Any analyst joining the team can clone and run it in minutes.

### How this maps to the target ways of working

| What a Data Centre of Excellence cares about | Where it shows up in this project                                           |
|----------------------------------------------|------------------------------------------------------------------------------|
| Enterprise BI platform (Power BI)            | `powerbi_export.py` — star schema, DAX templates, model JSON                 |
| Scalable, maintainable data products         | Four-gate pipeline, centralised config, pytest coverage, star schema design  |
| Reliability & documentation                  | Per-run JSON reports, structured logs, `docs/data_model.md`, inline module docs |
| Lifecycle management                         | GitHub Actions (lint → test → data validation), Makefile, devcontainer      |
| Data storytelling                            | Streamlit dashboard with five analysis modules, Plotly visuals, CSV export   |
| Statistical rigour                           | t-tests, Cohen's d, correlation matrices, Z-score outlier detection          |

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

**Live demo:** https://rejusam-clinicaldataanalysis-niv-streamlit-app-mggky8.streamlit.app/
**Script:** `niv_streamlit_app.py` | Launch locally: `streamlit run niv_streamlit_app.py`

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

The `data-validation` job runs only after `lint-and-test` passes, so no CSV lands on `main` without the processing code that produced it having been tested first.

---

## Lifecycle & Ops

This section describes how I'd run the project as a living data product — how changes flow through, how failures are surfaced, and how I'd roll back. It's written for the next analyst who inherits it.

### How a schema change flows through the four gates

Suppose the source starts emitting a new vital sign (e.g. `end_tidal_co2`) and a stakeholder wants it on the dashboard.

| Gate | What happens                                                                                  | What I check                                                                  |
|------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **0 — Config** (before Gate 1) | Add the new column name mapping in `mimic_waveform_processor.py` and any threshold in `config.py`. | `pytest tests/` locally — the standardisation tests catch column rename regressions. |
| **Gate 1 — Source validation** | The new column is detected in raw records. No change to the gate itself.                      | Logs still show the patient directories discovered.                           |
| **Gate 2 — ETL processing**    | The processor picks up the new column, applies quality flags, and writes it to the vitals CSV. | `data_pipeline.py --skip-etl` reads the updated CSV and re-runs downstream.   |
| **Gate 3 — Quality validation**| Completeness and range checks run on the new column if a threshold is defined in `config.py`. | Quality gate returns `PASS` / `WARN`; `WARN` is non-blocking but surfaced in the report. |
| **Gate 4 — Power BI export**   | `powerbi_export.py` adds the new column to the fact table keep-list and (optionally) a new DAX measure. | Row counts in the export summary match the processed CSVs; `data_model.json` reflects the new column. |

The pipeline is designed so that **a failure in any gate blocks the downstream gates**, and the JSON run report (`data/logs/pipeline_report_YYYYMMDD_HHMMSS.json`) names exactly which gate failed and why.

### Rollback strategy

The processed CSVs and Power BI exports are committed to git alongside the code that produced them. Rollback is therefore a single action:

```bash
# Revert to the last known-good snapshot
git checkout <good-commit> -- data/ tests/

# Re-verify against the reverted data
python data_pipeline.py --skip-etl
pytest tests/ -v
```

Because `.github/workflows/ci.yml` runs the data-validation step on every push, any commit that lands on `main` has already been validated against the CSVs it contains. This means **"the last green `main` commit"** is always a valid rollback target.

### Where the logs live

| Artifact                                           | Location                                     | Retention                                  |
|----------------------------------------------------|----------------------------------------------|--------------------------------------------|
| Per-run pipeline logs (timestamped text)           | `data/logs/pipeline_YYYYMMDD_HHMMSS.log`     | Kept locally; gitignored for privacy       |
| Per-run machine-readable reports (status + steps)  | `data/logs/pipeline_report_YYYYMMDD_HHMMSS.json` | Kept locally; gitignored                |
| CI run logs (lint, test, data validation)          | GitHub Actions → Actions tab on the repo    | GitHub default (~90 days)                  |
| Data model documentation                           | `data/powerbi_exports/data_model.json`       | Regenerated every export                   |

### Observability signals the panel would ask about

- **Did the pipeline finish?** `overall: "SUCCESS"` in the JSON report.
- **Which gate flagged something?** Each step in the JSON report has `status` (`PASS` / `WARN` / `SKIP` / `FAIL`) and a `details` string.
- **Why did a gate warn?** The `details` field names the column and the failing threshold (e.g. `"spo2 completeness 94.2% < target"`).
- **Are the processed CSVs still schema-valid?** The `data-validation` CI job asserts this on every push — a red CI run is the single source of truth.

### Things I would add next if this graduated from portfolio to production

- Replace committed CSVs with a lakehouse-backed table (Delta / Iceberg) so rollback becomes time-travel rather than git checkout.
- Move the JSON run reports into a log aggregator (Splunk, Log Analytics, OpenSearch) with alerting on `overall: "FAILED"`.
- Wire the Power BI dataset refresh to the pipeline success event so the report can't be ahead of the data that produced it.
- Promote the CI data-validation job into a nightly scheduled run against the live source, not just a PR gate.

---

## Project Structure

```
ClinicalDataAnalysis/
|-- .github/workflows/ci.yml      # CI/CD pipeline
|-- .devcontainer/devcontainer.json # Codespaces / Dev Container setup
|-- docs/
|   +-- data_model.md              # Star schema, grain, SCD decisions
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
