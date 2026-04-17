# AboutDataset — sample CSVs in `tests/`

This folder holds two CSV files for manual runs of the full AutoML Arena pipeline (**EDA → parallel models → evaluation → debate → judge**) via the **web UI** or **`POST /automl-debate`** / **`POST /api/v1/debate`**.

## Quick reference

| File | Typical `target_column` |
|------|-------------------------|
| [`Adult_income_dataset.csv`](./Adult_income_dataset.csv) | `income` |
| [`Telco-Customer-Churn.csv`](./Telco-Customer-Churn.csv) | `Churn` |

Below: provenance, key facts, and usage for each file.

---

## Files

### 1. `Adult_income_dataset.csv`

**Adult Income Dataset** (also called the **Census Income Dataset**) — A benchmark dataset from the **1994 US Census** used to predict whether a person earns **more than $50,000 per year**. Originally extracted and donated to the **UCI Machine Learning Repository** in **1996** by **Barry Becker** and **Ronny Kohavi**, it remains one of the most studied datasets in machine learning and algorithmic-fairness research.

**Key facts**

| | |
|--|--|
| **Source** | 1994 US Census database |
| **Instances** | 48,842 (full) / 32,561 (training subset) — *this repo’s CSV contains the full 48,842 rows* |
| **Features** | 14 demographic and employment attributes + target label (15 columns total in this file) |
| **Target classes** | `>50K` and `<=50K` annual income (column name: **`income`**) |
| **License** | CC BY 4.0 (UCI Machine Learning Repository) |

Use **`income`** as `target_column` in the app or API; values match the header exactly (`<=50K`, `>50K`).

### 2. `Telco-Customer-Churn.csv`

**Telco Customer Churn Dataset** — A publicly available sample dataset provided by **IBM** for predictive analytics and machine learning demonstrations. It models customer attrition in a telecommunications company, making it a common educational and benchmarking resource for churn prediction and classification techniques.

**Key facts**

| | |
|--|--|
| **Provider** | IBM Sample Data Assets |
| **Records** | 7,043 customer entries |
| **Target variable** | `Churn` (Yes / No) — use this as `target_column` in the app or API |
| **Features** | 21 customer, account, and service attributes |
| **Typical use** | Classification and feature-engineering practice |

In this repo’s CSV, the label column name is **`Churn`** (match the header exactly).

---

## Usage

**Web UI:** Start the backend and frontend (see [HOW_TO_RUN.md](../HOW_TO_RUN.md) at the repository root), open the app, choose a file from this folder, set the target column as above, and run.

**curl** (from the repository root):

```bash
curl -s -X POST "http://127.0.0.1:8000/automl-debate" \
  -F "file=@tests/Telco-Customer-Churn.csv" \
  -F "target_column=Churn"
```

Replace the file path and `target_column` for `Adult_income_dataset.csv` / `income` as needed.

For architecture, API shapes, and project setup, see [README.md](../README.md) at the repository root.
