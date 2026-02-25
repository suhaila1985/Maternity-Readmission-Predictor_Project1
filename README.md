# 🏥 Maternity Patient Readmission Prediction System

A **fairness-aware machine learning system** for predicting 30-day hospital readmission risk in maternity patients.

---

## 🎯 Key Achievement

**81.5% accuracy** while prioritising **ethical AI** — delivery type is deliberately excluded to prevent discrimination.

---

## 📂 Project Files

| File | Size | Description |
|------|------|-------------|
| `readmission_model.ipynb` | ~34 KB | Complete Jupyter notebook (Tasks 1–8 + ML + Fairness Audit) |
| `streamlit_app.py` | ~16 KB | Interactive prediction dashboard |
| `ethics_audit_report.pdf` | ~9.4 KB | Full fairness & ethics audit report |
| `maternity_data.csv` | ~20 KB | Training dataset (463 cleaned records) |
| `README.md` | ~4 KB | This file |
| `DEPLOYMENT_GUIDE.md` | ~5 KB | Deployment & setup instructions |
| `requirements.txt` | <1 KB | Python dependencies |

---

## 📊 Dataset Overview

- **Original records:** 500 patients
- **After cleaning:** 463 patients (37 removed — impossible values)
- **Columns:** PatientID, Age, DeliveryType, LaborDuration, Location, Complications, LengthofStaydays, Readmitted
- **Readmission rate:** 25.3% (117 readmitted, 346 not)

### Quality Issues Resolved

| Issue | Count | Action |
|-------|-------|--------|
| Age < 18 or > 45 | 22 | Removed |
| LOS < 2 days | 9 | Removed |
| Missing LaborDuration | 26 | Filled with median |
| Missing Age | 25 | Filled with median |
| Missing Complications | 11 | Filled with mode |

---

## 🤖 Model Summary

- **Algorithm:** Random Forest Classifier (100 trees, max depth 10)
- **Features:** Age, LaborDuration, LengthofStaydays, Location, Complications
- **Excluded:** DeliveryType (fairness decision — prevents discrimination)

### Performance

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | ~83% | **81.5%** |
| AUC | ~0.89 | **0.87** |
| Sensitivity | ~74% | 72% |
| Specificity | ~86% | 85% |

### Feature Importance

1. Length of Stay — 38%
2. Complications — 32%
3. Labor Duration — 18%
4. Location — 8%
5. Age — 4%

---

## ⚖️ Fairness

**Principle:** Individual Fairness — similar patients get similar risk scores.

**Bias audit results:**
- Delivery type accuracy gap: 2% ✅ (< 10% threshold)
- Location accuracy gap: 2% ✅ (< 10% threshold)
- **No significant bias detected**

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open: http://localhost:8501

### Option 2: Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook readmission_model.ipynb
```

### Option 3: Streamlit Cloud (Free)

1. Push files to GitHub
2. Visit https://streamlit.io/cloud
3. Click "New App" → select your repo and `streamlit_app.py`
4. Live in ~2 minutes

---

## ⚠️ Deployment Checklist (Before Clinical Use)

- [ ] IRB / Ethics Committee approval
- [ ] Explicit informed consent protocol
- [ ] Data encryption (AES-256)
- [ ] Local hospital data validation
- [ ] Clinical staff training
- [ ] Monitoring & alerting setup
- [ ] Adverse event reporting workflow

---

## 📋 Notebook Contents (Tasks 1–8)

| Task | Description |
|------|-------------|
| Task 1 | Basic statistics (mean, median, std) |
| Task 2 | Delivery type counts & distribution |
| Task 3 | Readmission rates overall & by subgroup |
| Task 4 | Comparisons by delivery type |
| Task 5 | Histograms — Age, Labor Duration, LOS |
| Task 6 | Data prep + bar & pie charts |
| Task 7 | Cross-tabulations + LOS Paradox |
| Task 8 | Quality validation report |
| ML | Feature engineering + model training |
| Fairness | Bias detection + fairness audit |

---

**Version:** 1.0 | **Date:** February 2024 | **License:** Educational & Research Use
