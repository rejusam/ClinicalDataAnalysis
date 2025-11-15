# Respiratory Biometric Data Analysis Platform

This project demonstrates a full pipeline for clinical data analysis, from raw data processing to an interactive web-based dashboard. he analysis focuses on **respiratory biometric parameters** from critically ill ICU patients, evaluating oxygen saturation (SpO2), respiratory rate, heart rate, and hypoxemia episodes.

---

## 🗂️ Project Structure

```
respiratory-biometric-analysis/
│
├── data/
│   ├── mimic3wdb/
│   │   └── p00/                          # Raw MIMIC-III waveform data
│   │       ├── p000020/
│   │       ├── p000030/
│   │       └── ...
│   ├── mimic_waveform_vitals.csv         # Processed time-series data
│   └── mimic_waveform_summary.csv        # Patient-level summary statistics
│
├── get_data_data.sh                      # Data download script
├── process_mimic_waveforms.py            # Data processing pipeline
├── niv_streamlit_app.py                  # Interactive dashboard
└── README.md                             # This file
```
---

## Complete Workflow

### Phase 1: Data Acquisition

**Script:** `get_data_data.sh`

Downloads raw waveform data from the MIMIC-III Waveform Database Matched Subset via PhysioNet. The script retrieves 10 patient records from the p00/ directory, each containing:

- **Waveform records** (*.hea, *.dat): High-frequency physiological signals
- **Numerics records** (*n.hea, *n.dat): Minute-by-minute vital signs

**Patients Downloaded:**
- p000020, p000030, p000033, p000052, p000079
- p000085, p000123, p000154, p000208, p000262

**Data Volume:** ~500MB-1GB depending on monitoring duration

```bash
bash get_data_data.sh
```

---

### Phase 2: Data Processing & Feature Engineering

**Script:** `process_mimic_waveforms.py`

A comprehensive ETL (Extract, Transform, Load) pipeline that processes raw waveform data into analysis-ready datasets.

#### Key Processing Steps:

1. **Data Extraction**
   - Reads MIMIC-III waveform files using the WFDB Python library
   - Extracts numerics records containing minute-by-minute vital signs
   - Handles multiple recording sessions per patient

2. **Variable Standardization**
   - Maps inconsistent monitor nomenclature to standardized variable names
   - Handles variations across different ICU monitoring systems:
     - `HR`, `PULSE` → `heart_rate`
     - `SpO2`, `SAO2`, `%SpO2` → `spo2`
     - `RESP`, `RR` → `respiratory_rate`
   - Consolidates blood pressure measurements (invasive and non-invasive)

3. **Feature Engineering**
   - **Hypoxemia Detection**: Flags SpO2 < 90% events
   - **Tachypnea/Bradypnea**: Identifies abnormal respiratory rates (>24 or <12 bpm)
   - **SpO2 Variability**: Rolling window standard deviation (10-minute window)
   - **Temporal Metrics**: Hours from ICU admission for trend analysis

4. **Data Quality Procedures**
   - **Range Validation**: Checks physiologically plausible ranges
   - **Outlier Detection**: Z-score methodology (|Z| > 3)
   - **Completeness Assessment**: Missing data quantification
   - **Quality Flags**: Labels suspicious or out-of-range values

5. **Patient-Level Aggregation**
   - Calculates summary statistics per patient:
     - Mean, SD, min/max for each vital sign
     - Hypoxemia burden (% time SpO2 < 90%)
     - Tachypnea episodes count
     - Total monitoring duration

#### Output Files:

**`mimic_waveform_vitals.csv`** (34,630 rows × 79 columns)
- Time-series dataset with minute-by-minute measurements
- Key columns: `subject_id`, `hours`, `spo2`, `respiratory_rate`, `heart_rate`, `hypoxemia`, `tachypnea`
- Includes derived metrics and quality flags

**`mimic_waveform_summary.csv`** (10 rows × 14 columns)
- Patient-level summary with aggregated statistics
- Key columns: `subject_id`, `monitoring_hours`, `spo2_mean`, `spo2_std`, `hypoxemia_rate`, `outcome`

```bash
python process_mimic_waveforms.py
```

**Processing Statistics:**
- Patients processed: 10
- Total measurements: 34,630
- Average monitoring: 21.9 hours/patient
- Data completeness: >90% for key variables

---

### Phase 3: Interactive Analysis Dashboard

**Script:** `niv_streamlit_app.py`

A comprehensive web-based analytical platform with five integrated analysis modules:

#### 📊 Module 1: Overview & Demographics
- **Patient Cohort Characteristics**
  - Sample size and demographics
  - Monitoring duration distribution
  - Severity indicators (APACHE II surrogate)
  
- **Respiratory Severity Metrics**
  - Mean SpO2 distribution across patients
  - Hypoxemia burden visualization
  - Respiratory rate variability

- **Data Overview Tables**
  - Patient summary statistics
  - Raw time-series data preview

#### 🫀 Module 2: Respiratory Biometrics
- **Vital Signs Distribution Analysis**
  - SpO2 histogram with clinical thresholds (>92% target)
  - Respiratory rate distribution with tachypnea/bradypnea zones
  - Heart rate patterns
  
- **Patient-Level Comparisons**
  - Individual patient SpO2 performance
  - Hypoxemia burden by patient
  
- **Biometric Relationships**
  - SpO2 variability vs mean SpO2 scatter plot
  - Outcome correlation with respiratory metrics

#### 📈 Module 3: Temporal Analysis
- **Individual Patient Trajectories**
  - Time-series plots: SpO2, respiratory rate, heart rate
  - Hypoxemia episode detection and visualization
  - Clinical event identification
  
- **Aggregate Trends**
  - Hourly average trends across all patients
  - Standard deviation bands for uncertainty quantification
  - Time-to-event analysis

#### 🔍 Module 4: Data Quality Report
- **Completeness Analysis**
  - Variable-level completeness rates
  - Target threshold: >95% completeness
  - Visual indicators (green/orange/red)
  
- **Quality Flags Distribution**
  - Valid records percentage
  - Out-of-range values count
  - Suspicious measurements flagging
  
- **Outlier Detection**
  - Z-score based methodology (|Z| > 3)
  - Visual identification in scatter plots
  - Percentage of outliers quantified
  
- **Validation Summary**
  - GCP-compliant checklist
  - Range validation status
  - Data integrity confirmation

#### 📋 Module 5: Statistical Analysis
- **Hypothesis Testing**
  - Independent t-tests for outcome comparisons
  - Effect size calculations (Cohen's d)
  - Statistical significance interpretation
  
- **Correlation Analysis**
  - Heatmap of respiratory parameter correlations
  - Key relationship identification
  - Strength and direction of associations
  
- **Clinical Outcome Analysis**
  - Hypoxemia burden vs outcome
  - Respiratory rate vs SpO2 relationships
  - Survival analysis considerations

```bash
streamlit run niv_streamlit_app.py
```

**Dashboard Features:**
- 📱 Responsive design for all screen sizes
- 💾 CSV download functionality for reports
- 🎨 Color-coded visualizations for clinical thresholds
- 📊 Interactive Plotly charts (zoom, pan, hover details)
- 🔄 Real-time filtering and patient selection

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- ~2GB free disk space

**Required packages:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scipy>=1.11.0
wfdb>=4.1.0
```


### Download Data

```bash
bash get_data_data.sh
```

This downloads 10 patient records (~500MB-1GB) from MIMIC-III Waveform Database.


### Launch Dashboard

```bash
streamlit run niv_streamlit_app.py
```

Dashboard opens automatically at: `http://localhost:8501`

---

### Key Variables Analyzed

**Primary Respiratory Parameters:**
- `spo2`: Oxygen saturation (%)
- `respiratory_rate`: Breaths per minute
- `heart_rate`: Beats per minute

**Derived Metrics:**
- `hypoxemia`: SpO2 < 90% flag
- `tachypnea`: Respiratory rate > 24 bpm flag
- `spo2_variability`: Rolling SD of SpO2
- `hypoxemia_rate`: % time in hypoxemia per patient

**Hemodynamic Parameters:**
- `ABPSys/ABPDias/ABPMean`: Arterial blood pressure
- `NBPSys/NBPDias/NBPMean`: Non-invasive blood pressure
- `CVP`: Central venous pressure

---

## Clinical Research Applications

This pipeline can be adapted for:

- **NIV vs Invasive Ventilation Studies:** Compare respiratory support outcomes
- **Predictive Modeling:** Early warning systems for respiratory deterioration
- **Protocol Development:** Evidence-based ventilation protocols
- **Quality Improvement:** ICU monitoring quality assessments
- **Phenotype Discovery:** Identify respiratory failure subtypes

---

## 📚 References

1. **MIMIC-III Waveform Database Matched Subset**  
   Moody, B., Moody, G., Villarroel, M., Clifford, G., & Silva, I. (2020).  
   PhysioNet. DOI: 10.13026/C2294B

2. **MIMIC-III Clinical Database**  
   Johnson, A. E. W., et al. (2016). MIMIC-III, a freely accessible critical care database.  
   *Scientific Data*, 3, 160035.

3. **ICH E9 Statistical Principles for Clinical Trials**  
   International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use.

4. **ICH E6(R3) Good Clinical Practice**  
   International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use.

---

## 🙏 Acknowledgments

- PhysioNet for providing open access to clinical data
- MIMIC-III research team at MIT Laboratory for Computational Physiology
- Open-source community for Python scientific computing tools

---

**⭐ If this project helped you, please consider giving it a star!**

---

*Last Updated: November 2025*
