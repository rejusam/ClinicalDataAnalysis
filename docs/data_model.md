# Data Model — ICU Respiratory Biometrics

A plain-English walkthrough of the Power BI star schema: grain, keys, SCD
decisions, pre-aggregates, and the design trade-offs behind them.

This document is the reference I use when explaining the model to a
stakeholder or reviewing it with another analyst.

---

## 1. Model at a glance

```
                    dim_patient
                        |
   dim_time -------- fact_vitals -------- dim_quality
```

Classic star schema. One fact, three conformed dimensions, no snowflaking.
All relationships are **many-to-one**, single direction (dim filters fact),
which is the Power BI default and keeps measure behaviour predictable.

| Table            | Type      | Grain                              | Keys                         |
|------------------|-----------|------------------------------------|------------------------------|
| `fact_vitals`    | Fact      | One row per patient per minute     | `patient_key`, `hour_key`, `quality_key` |
| `dim_patient`    | Dimension | One row per patient                | `patient_key` (surrogate)    |
| `dim_time`       | Dimension | One row per hour of ICU stay       | `hour_key` (surrogate)       |
| `dim_quality`    | Dimension | One row per quality classification | `quality_key` (surrogate)    |

---

## 2. Fact table — `fact_vitals`

**Grain:** One row per patient per minute of ICU monitoring.
**Row count at current volume:** ~34,600.

### Columns

| Column            | Role          | Notes                                               |
|-------------------|---------------|-----------------------------------------------------|
| `patient_key`     | FK            | Links to `dim_patient`                              |
| `hour_key`        | FK            | Links to `dim_time` (rounded from minute timestamp) |
| `quality_key`     | FK            | Links to `dim_quality`                              |
| `heart_rate`      | Measure       | bpm                                                 |
| `respiratory_rate`| Measure       | breaths/min                                         |
| `spo2`            | Measure       | % oxygen saturation                                 |
| `hypoxemia`       | Flag (0/1)    | Derived: `spo2 < 90`                                |
| `tachypnea`       | Flag (0/1)    | Derived: `respiratory_rate > 25`                    |
| `spo2_variability`| Measure       | 10-minute rolling std of `spo2`                     |

### Grain vs. time dimension — a deliberate trade-off

The fact stores **minute-level** rows but joins to `dim_time` at **hour**
granularity. This is intentional:

- **Why minute-level in the fact:** clinical measures like hypoxemia
  variability and rolling SpO2 only make sense at the raw sampling rate.
  Aggregating to hour up front would erase the signal the dashboard needs.
- **Why hour-level time dimension:** stakeholders filter by "first 24h",
  "day shift vs. night shift", "day of stay" — none of which need sub-hour
  resolution. Smaller `dim_time` = faster filter propagation.
- **How it stays honest:** the fact keeps its native minute grain; joining
  at hour is a filter operation, not an aggregation. Minute-level detail is
  recoverable via measures that use the full row set.

The `agg_hourly_vitals` table (see §5) exists so most report visuals don't
touch the minute-level fact at all.

---

## 3. Dimension tables

### `dim_patient`

**Grain:** One row per patient.

Surrogate `patient_key` (1..N) is assigned at build time rather than reusing
the source `subject_id`, so the schema is not coupled to MIMIC naming.

Attributes include `ventilation_type`, `monitoring_hours`, `spo2_mean`,
`hypoxemia_rate`, and a derived `risk_category` (`Low` / `Moderate` / `High`)
bucketed from hypoxemia rate using clinically informed cut-points
(10%, 30%).

**SCD decision — Type 1.** Patient attributes here are computed summaries
from the ICU stay, not mutable master data. If hypoxemia rate is
recalculated after a fresh ETL run, the old value is overwritten. No history
is preserved because no stakeholder has asked "what did we think the risk
category was last month" — only "what is it now". Revisit this if the model
moves from a one-shot portfolio build to a live feed of incoming patients.

### `dim_time`

**Grain:** One row per hour of ICU stay (not calendar time).

This is the non-obvious call: `dim_time` is **relative**, not **absolute**.
Keys are hours since ICU admission (`hour_of_stay`), not wall-clock
timestamps. Reasons:

- The analytical question is "what happens to patients in their first 24h",
  not "what happened at 3pm on Tuesday". Cohort comparisons need relative
  time.
- Wall-clock time is meaningless across patients with different admission
  dates unless the study is about staffing patterns.
- It keeps the dimension tiny (one row per hour of stay, max ~72 in this
  dataset) so slicers are instant.

Derived attributes: `shift` (Day/Night using 07:00–19:00 split on
`hour_of_stay % 24`), `day_of_stay`, and a `period` bucket
(`First 24h` / `24-48h` / `48-72h`). These are the slicers the dashboard
actually uses.

**SCD decision — Type 1.** The dimension is regenerated on each run from
the max observed hour. There is no attribute history to preserve.

### `dim_quality`

**Grain:** One row per quality classification — `Valid`, `Out_of_range`,
`Suspicious`.

Exists so slicers can toggle between "all rows" and "rows safe for
analysis" without embedding that logic in every measure. `is_valid_for_analysis`
is a boolean attribute on the dim that measures can reference.

**SCD decision — Type 1, effectively immutable.** The list is fixed by the
quality rules in `mimic_waveform_processor.py`. If a new flag is added,
it's a schema change, not an SCD event.

---

## 4. Keys and referential integrity

- All dimensional keys are **surrogate integers** assigned at build time,
  not natural keys from the source. This insulates the model from source
  system naming (`subject_id`) and keeps joins on small integer columns.
- `patient_key` is assigned via a dict map in `powerbi_export.py`:
  `range(1, N+1)` over the patient summary table. Deterministic for a given
  input ordering.
- `quality_key` is assigned from a fixed lookup (`Valid=1, Out_of_range=2,
  Suspicious=3`). Rows missing a flag fall back to `Valid` and are logged.
- `hour_key` is derived (`round(hours, 0)`) rather than looked up; the join
  succeeds because `dim_time` enumerates every hour from 0 to `max(hours)`.
- **No orphan rows.** The export fails closed if any fact row can't resolve
  its FKs — this is what the `.fillna(1).astype(int)` guard is protecting
  against for quality, and what the enumerated `dim_time` guarantees for
  time.

---

## 5. Pre-aggregates — `agg_hourly_vitals`

**Grain:** One row per patient per hour.

Not strictly part of the star, but shipped alongside it. Contains mean /
min / max / std of SpO2, heart rate, and respiratory rate per hour, plus
total hypoxemia minutes.

**Why it exists:** most report visuals (trend lines, shift summaries,
cohort comparisons) are hourly rollups. Hitting the minute-level fact for
every visual would scale badly as patients are added. Power BI users point
their trend visuals at this table and only drop to `fact_vitals` when they
need the raw signal.

**Trade-off:** two tables means two refreshes, and care is needed to keep
measures consistent between them. The alternative — aggregation-aware
modelling inside Power BI itself — is the next upgrade if this moves to
Premium capacity.

---

## 6. DAX measures — design notes

All measures live in `dax_measures.json` as starter templates. A few design
choices worth calling out:

- **`Average SpO2`** uses `AVERAGE(fact_vitals[spo2])` rather than a
  summary table. Averaging at the fact keeps filter context honest — any
  slicer the user applies propagates through.
- **`Hypoxemia Rate %`** is `DIVIDE(filtered_count, total_count)` rather
  than a pre-computed column. Pre-computing would freeze the denominator
  at the wrong grain; doing it in DAX keeps it responsive to slicer
  context.
- **`High Risk Patient Count`** filters `dim_patient`, not `fact_vitals`.
  The fact has 34K rows and repeats `risk_category` per patient; filtering
  the dim is one join, not thousands.
- `DIVIDE` is used over `/` everywhere — it returns `BLANK()` on
  divide-by-zero instead of an error, which is the right default for a
  self-service report where a slicer might empty the denominator.

---

## 7. Known limitations

- **Patient cohort is small** (10 patients from MIMIC-III matched subset).
  The model is built to scale — grain, keys, and relationships don't
  change at 10K patients — but cardinality assumptions in the dimensions
  should be re-checked at that point.
- **No time-series dimension on wall clock.** If the analytical question
  ever becomes "what happened on Tuesday evenings", `dim_time` needs an
  absolute-time counterpart.
- **SCD Type 1 everywhere.** Appropriate for the current use case
  (analytics on a closed dataset), wrong for a live operational feed.
- **Pre-aggregate consistency is enforced by convention, not by the model.**
  A query-rewrite layer (Power BI aggregations, or dbt `sources` +
  `metrics`) would be the proper fix if this graduates to production.

---

## 8. How this maps to the code

| Concept                     | File                               |
|-----------------------------|------------------------------------|
| Dim/fact construction       | `powerbi_export.py`                |
| Grain and quality flags     | `mimic_waveform_processor.py`      |
| Risk-category cut-points    | `powerbi_export.py::create_dim_patient` |
| Shift definition (Day/Night)| `powerbi_export.py::create_dim_time`    |
| Referential integrity guards| `powerbi_export.py::create_fact_vitals` |
| Model documentation output  | `powerbi_export.py::generate_data_model_doc` |
| Generated model JSON        | `data/powerbi_exports/data_model.json`       |
