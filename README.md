# NIV Clinical Data Analysis Portfolio Project

This project demonstrates a full pipeline for clinical data analysis, from raw data processing to an interactive web-based dashboard. It uses a simulated dataset based on the MIMIC-III Waveform Database to analyze and compare outcomes for patients on different types of respiratory support.

This project is intended to showcase skills in:
-   Clinical data programming (Python, Pandas)
-   Biometric and time-series data analysis
-   Data quality and validation
-   Statistical analysis and hypothesis testing
-   Creating interactive data visualizations (Streamlit)

---

## Project Structure

-   `mimic_waveform_processor.py`: A Python script that processes raw, simulated waveform data. It cleans the data, standardizes column names, handles inconsistencies, and generates a patient-level summary dataset.
-   `niv_streamlit_app.py`: An interactive Streamlit web application for visualizing the processed data. It includes modules for demographic overviews, biometric analysis, clinical outcomes, data quality reporting, and statistical tests.
-   `data/`: This directory holds the processed data.
    -   `mimic_waveform_vitals.csv`: Time-series data with vital signs for each patient.
    -   `mimic_waveform_summary.csv`: A summary file with one row per patient, containing aggregated metrics and outcomes.

---

## How to Run This Project

### 1. Generate the Analysis Data

First, you must run the data processor to generate the clean CSV files required by the Streamlit application.

```bash
python mimic_waveform_processor.py
```
This will create `mimic_waveform_vitals.csv` and `mimic_waveform_summary.csv` in the `data/` directory.

### 2. Launch the Interactive Dashboard

Once the data files have been generated, you can run the Streamlit application.

```bash
streamlit run niv_streamlit_app.py
```

This will launch the web application in your browser, where you can explore the different analysis modules.
