# 🚀 HAYNES 282 DIGITAL TWIN

## AI-Powered LPBF Process Intelligence Platform for Additive Manufacturing

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/ML-XGBoost%20%7C%20CatBoost-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Domain-Additive%20Manufacturing-black?style=for-the-badge">
</p>

---

# 📌 Table of Contents

1. Project Overview
2. Why This Project Matters
3. Key Features
4. Digital Twin Workflow
5. Machine Learning Models
6. Project Structure
7. Installation Guide
8. How to Run
9. How to Use
10. Defect Prediction Logic
11. Mechanical Property Outputs
12. Inverse Optimization Module
13. Technology Stack
14. Performance & Validation
15. Future Scope
16. Research Applications
17. Author

---

# 📖 Project Overview

This project is an **AI-based Digital Twin Platform** for **Laser Powder Bed Fusion (LPBF)** manufacturing of **Haynes 282 Nickel Superalloy**.

It combines **Machine Learning**, **Manufacturing Science**, and **Interactive UI Design** to predict process outcomes instantly using only printing parameters.

The system helps users choose better process parameters before manufacturing starts.

---

# 🏭 Why This Project Matters

Laser Powder Bed Fusion involves many variables such as power, speed, hatch spacing, and layer thickness.

Wrong settings can cause:

❌ Lack of Fusion <br>
❌ Keyhole Porosity<br>
❌ Poor Mechanical Strength<br>
❌ Melt Pool Instability<br>
❌ Expensive Failed Builds

This Digital Twin solves that problem by predicting outcomes before printing.

---

# 🎯 Core Objectives

✅ Predict print quality in real time<br>
✅ Predict melt pool geometry<br>
✅ Predict mechanical properties<br>
✅ Reduce trial-and-error experimentation<br>
✅ Enable AI-driven process optimization<br>
✅ Build a research-grade smart manufacturing platform

---

# ⚙️ User Inputs

The platform takes four LPBF process parameters:

| Parameter       | Symbol | Unit |
| --------------- | ------ | ---- |
| Laser Power     | P      | W    |
| Scan Speed      | v      | mm/s |
| Hatch Spacing   | h      | mm   |
| Layer Thickness | t      | mm   |

---

# 🧠 AI Outputs (7 Trained Models)

The platform predicts the following outputs instantly:

| Category     | Predicted Output       |
| ------------ | ---------------------- |
| Defect State | LOF / Stable / Keyhole |
| Strength     | Yield Strength         |
| Strength     | UTS                    |
| Hardness     | Hardness HRA           |
| Ductility    | Elongation             |
| Melt Pool    | Width                  |
| Melt Pool    | Depth                  |

---

# 🔁 Digital Twin Workflow

```text
User Inputs (P, v, h, t)
        ↓
7 Trained ML Models
        ↓
Quality + Geometry + Mechanical Predictions
        ↓
Recommendations / Optimization
```

---

# 🤖 Machine Learning Models Used

| File Name             | Purpose                   |
| --------------------- | ------------------------- |
| defect_classifier.pkl | Defect classification     |
| YieldStrength_MPa.pkl | Yield strength prediction |
| UTS_MPa.pkl           | Ultimate tensile strength |
| Hardness_HRA.pkl      | Hardness prediction       |
| Elongation_pct.pkl    | Elongation prediction     |
| MeltPoolWidth_um.pkl  | Melt pool width           |
| MeltPoolDepth_um.pkl  | Melt pool depth           |
| feature_order.pkl     | Input feature order       |

---

# 📁 Project Structure

```bash
Haynes282-DigitalTwin/
│── app.py
│── requirements.txt
│── README.md
│── pages
│   │── 1_Forward_Prediction.py
│   ├── 2_Inverse_optimizer.py
│
└── models/
    ├── defect_classifier.pkl
    ├── YieldStrength_MPa.pkl
    ├── UTS_MPa.pkl
    ├── Hardness_HRA.pkl
    ├── Elongation_pct.pkl
    ├── MeltPoolWidth_um.pkl
    ├── MeltPoolDepth_um.pkl
    └── feature_order.pkl
```

---

# 💻 Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/Saatu23/Haynes282-DigitalTwin.git
cd Haynes282-DigitalTwin
```

---

## Step 2 — Create Virtual Environment

```bash
python -m venv venv
```

### Activate

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

# 🖥️ How to Use

## Step 1

Enter process parameters using sidebar sliders:

* Power
* Speed
* Hatch Spacing
* Layer Thickness

## Step 2

Click **Predict**

## Step 3

View outputs:

✅ Defect state<br>
✅ Melt pool geometry<br>
✅ Yield strength<br>
✅ UTS<br>
✅ Hardness<br>
✅ Elongation<br>
✅ Charts & recommendations

---

# ⚠️ Defect Classification Logic

| Prediction | Meaning                | Recommended Action            |
| ---------- | ---------------------- | ----------------------------- |
| 🟢 Stable  | Good processing window | Continue                      |
| 🟠 LOF     | Insufficient fusion    | Increase power / reduce speed |
| 🔴 Keyhole | Excess energy input    | Reduce power / increase speed |

---

# 📊 Mechanical Property Predictions

The platform predicts:

| Property       | Meaning                           |
| -------------- | --------------------------------- |
| Yield Strength | Resistance to plastic deformation |
| UTS            | Maximum tensile load capacity     |
| Hardness       | Surface / bulk hardness           |
| Elongation     | Ductility before fracture         |

---

# 🔥 Melt Pool Geometry Prediction

The platform predicts:

| Output          | Importance                |
| --------------- | ------------------------- |
| Melt Pool Width | Track overlap & stability |
| Melt Pool Depth | Fusion depth & bonding    |

---

# 🎯 Inverse Optimization Module

The project can be expanded to recommend best parameters automatically.

## Example Goal

```text
Target:
UTS > 1100 MPa
YS > 850 MPa
Low defect probability
Stable melt pool
```

## Output

```text
Recommended:
Best P, v, h, t
```

This transforms the project from prediction system to intelligent manufacturing assistant.

---

# 📈 Performance & Validation

The models were trained on structured process-property data using ensemble algorithms.

Typical expected performance:

| Task                  | Expected Performance |
| --------------------- | -------------------- |
| Defect Classification | 90%+ Accuracy        |
| Strength Prediction   | High R²              |
| Melt Pool Prediction  | High Accuracy        |
| Real-time Inference   | Instant              |

---

# 🛠 Technology Stack

| Layer           | Technology         |
| --------------- | ------------------ |
| Frontend        | Streamlit          |
| ML Algorithms   | XGBoost / CatBoost |
| Data Processing | Pandas / NumPy     |
| Visualization   | Plotly             |
| Model Storage   | Joblib             |

---

# 📚 Research Importance

This project demonstrates:

✅ Digital Twin for Additive Manufacturing<br>
✅ AI-driven Process Optimization<br>
✅ Process-Structure-Property Modeling<br>
✅ Smart Manufacturing Automation<br>
✅ Practical LPBF Decision Support<br>

Suitable for:

🎓 Thesis Project<br>
📄 Journal / Conference Paper<br>
🏭 Industrial Demonstration<br>
💼 Resume / Portfolio

---

# 🚀 Future Scope

Planned upgrades:

* Bayesian Optimization
* SHAP Explainability
* Multi-alloy Support
* Real Sensor Data Integration
* Cloud Deployment
* Production Dashboard
* Hybrid Physics + AI Modeling

---

# 🧪 Example Use Cases

### Aerospace Components

Optimize high-strength nickel alloy parts.

### Turbine Components

Improve temperature-resistant parts.

### Research Labs

Study LPBF process windows rapidly.

### Industry

Reduce machine trial costs.

---

# 👨‍💻 Author

**Satyam Mishra**
Mechanical Engineering | Additive Manufacturing | AI Research

---

# ⭐ Support This Project

If you found this project interesting:

⭐ Star the repository
🍴 Fork the repo
📢 Share with researchers

---

# 📌 Status

```text
Version : 1.0
State   : Production Ready
Domain  : LPBF Haynes 282 Digital Twin
```
