# P.R.I.M.A.
Premium Risk Indexing and Modelling for Agriculture

> **A data-driven framework for dynamic, explainable, and risk-sensitive crop insurance pricing.**

P.R.I.M.A. (**Premium Risk Indexing and Modelling for Agriculture**) is an AI-driven crop insurance risk assessment and dynamic premium adjustment system designed to address the limitations of static actuarial pricing.

The project combines **phenology-based risk modelling, machine learning, ensemble prediction, explainable AI, and multi-objective optimization** to estimate claim risk and generate individualized insurance premiums.

The current implementation focuses on **California almond orchards**, using climate, agronomic, and insurance-related variables to model evolving seasonal risk.

---

## Overview

Traditional crop insurance pricing often relies on historical averages and relatively static actuarial assumptions. However, agricultural risk changes throughout the growing season due to factors such as:

* Frost events
* Heat stress
* Rainfall
* Crop development stages
* Orchard characteristics
* Yield expectations
* Insurance coverage decisions

P.R.I.M.A. addresses this by incorporating **phenology-specific weather risks** into a machine learning pipeline and translating the resulting risk assessment into an interpretable premium recommendation.

The system is designed around three key questions:

1. **How likely is an insurance claim?**
2. **What factors are driving that risk?**
3. **How should the premium respond to the estimated risk?**

---

## Key Features

### 🌱 Phenology-Based Risk Modelling

Instead of treating the growing season as a single period, P.R.I.M.A. maps weather conditions to different stages of almond development.

Risk indicators include:

* Dormancy Frost Risk
* Bloom Frost Risk
* Bloom Rain Risk
* Fruit Growth Heat Risk
* Hull Split Heat Risk
* Harvest Rain Risk
* Growing Degree / Chill-related indicators

This allows the model to capture **when** a weather event occurs, rather than only whether it occurs.

---

### 🤖 Ensemble Claim Prediction

P.R.I.M.A. combines three machine learning models:

* **CART Decision Tree**
* **XGBoost**
* **Feedforward Neural Network**

Each model produces a probability of claim occurrence. The final ensemble probability is calculated by averaging the predictions:

$$
P_{claim} =
\frac{P_{CART}+P_{XGBoost}+P_{NN}}{3}
$$

This approach combines different model architectures to capture nonlinear relationships and interactions in agricultural risk data.

---

### 📊 Explainable AI

The system uses **SHAP (SHapley Additive exPlanations)** to identify the factors contributing most strongly to an individual risk prediction.

Rather than returning only a claim probability, P.R.I.M.A. provides an interpretable view of the major risk drivers.

This is particularly important in insurance applications where pricing decisions need to be understandable to both insurers and policyholders.

---

### ⚖️ Multi-Objective Optimization

P.R.I.M.A. uses **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** to optimize the weights assigned to phenology-specific risk indicators.

The optimization considers two competing objectives:

* **Insurer Satisfaction** — better identification and pricing of high-risk policies
* **Farmer Satisfaction** — fairer and more transparent pricing for lower-risk growers

A balanced solution is selected from the resulting Pareto frontier using an **Achievement Scalarization Function (ASF)**.

The resulting weights are used to calculate the **Viability Score**.

---

### 💰 Dynamic Premium Adjustment

The Viability Score summarizes the overall health and risk profile of an orchard.

Higher viability indicates lower aggregate risk, while lower viability indicates higher risk.

The application then combines:

* Estimated claim probability
* Expected loss
* Insured liability
* County-level actuarial benchmarks

to generate a dynamic premium recommendation.

The system also provides an **intra-seasonal adjustment framework**, allowing premiums to respond as additional weather and phenological information becomes available.

---

## System Architecture

```text
                 Farm & Insurance Inputs
                          │
                          ▼
              ┌──────────────────────┐
              │ Data Preprocessing    │
              │ & Feature Engineering │
              └──────────┬───────────┘
                         │
                         ▼
             Phenology-Based Risk Features
                         │
                         ▼
              ┌──────────────────────┐
              │ Machine Learning     │
              │                      │
              │ CART                 │
              │ XGBoost              │
              │ Neural Network       │
              └──────────┬───────────┘
                         │
                         ▼
                 Ensemble Probability
                         │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
             SHAP Analysis   Expected Loss
                  │             │
                  │             ▼
                  │      Dynamic Premium
                  │
                  ▼
             Risk Explanation
```

---

## Dataset

The research framework uses a dataset comprising **38,000 almond orchard insurance policies covering 2020–2024**.

The modelling framework integrates:

* Localized climate information from **California Irrigation Management Information System (CIMIS)**
* Insurance policy information structured around **USDA Risk Management Agency (RMA)** frameworks
* Farm-level agronomic and orchard characteristics

The research dataset contains approximately **3.7% claim observations**, creating a substantial class-imbalance problem.

To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied exclusively to the training data.

> **Note:** The insurance policy component of the research dataset is partially simulated, and therefore the results should not be interpreted as directly representing production insurance pricing.

---

## Model Performance

The models were evaluated using **F1-score** and **ROC-AUC**, with particular emphasis on F1 because of the highly imbalanced claim distribution.

| Model                       |    ROC-AUC | Avg. Precision-Recall |   F1-Score |
| --------------------------- | ---------: | --------------------: | ---------: |
| Logistic Regression         |     0.9247 |                0.3746 |     0.2987 |
| Random Forest               |     0.9108 |                0.3183 |     0.3578 |
| **P.R.I.M.A. DMA Ensemble** | **0.9206** |            **0.3568** | **0.4134** |

The ensemble achieved an **F1-score of 0.4134** and **ROC-AUC of 0.9206** on the evaluation data.

---

## Technology Stack

| Category          | Technologies          |
| ----------------- | --------------------- |
| Language          | Python                |
| Application       | Streamlit             |
| Data Processing   | Pandas, NumPy         |
| Machine Learning  | Scikit-learn, XGBoost |
| Deep Learning     | TensorFlow / Keras    |
| Explainability    | SHAP                  |
| Model Persistence | Joblib                |
| Optimization      | NSGA-II               |
| Visualization     | Matplotlib            |

The repository includes the trained models and preprocessing artifacts required by the Streamlit application.

---

## Application

P.R.I.M.A. is implemented as an interactive **Streamlit** application.

Users can provide information such as:

* County
* Almond variety
* Farming experience
* Tree age
* Orchard size
* Planting density
* Irrigation system
* Approved yield
* Coverage level
* Price election
* Forecasted chill units
* Frost days
* Precipitation
* Heat-stress days
* Vapour Pressure Deficit (VPD)

The application processes these inputs through the trained models and returns risk and pricing metrics.

### Example Outputs

The application generates:

* **Claim Probability**
* **Orchard Health Score**
* **Model Confidence**
* **Expected Loss**
* **Dynamic Premium**
* **Traditional County Premium**
* **Risk Drivers using SHAP**

---

## Project Structure

```text
P.R.I.M.A./
│
├── app.py
├── requirements.txt
│
├── training_data.csv
├── shap_background_data.csv
│
├── preprocessor.joblib
├── cart_model.joblib
├── xgb_model.joblib
├── nn_model.keras
│
├── definitive_weights.npy
└── context_data.joblib
```

### File Description

| File                       | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| `app.py`                   | Streamlit application and prediction pipeline |
| `training_data.csv`        | Training/reference dataset                    |
| `shap_background_data.csv` | Data used for SHAP-related processing         |
| `preprocessor.joblib`      | Saved preprocessing pipeline                  |
| `cart_model.joblib`        | Trained CART model                            |
| `xgb_model.joblib`         | Trained XGBoost model                         |
| `nn_model.keras`           | Trained neural network                        |
| `definitive_weights.npy`   | Optimized phenology weights                   |
| `context_data.joblib`      | Saved model/application configuration         |

The repository currently contains these model, data, and application artifacts.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aasthaaachhabra/P.R.I.M.A..git
cd P.R.I.M.A.
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project specifies Streamlit, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, SHAP, Joblib, and Matplotlib dependencies.

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Research Methodology

The complete research pipeline consists of four major stages:

### 1. Data & Feature Engineering

Climate and agronomic variables are aligned with almond phenological stages to construct targeted risk indicators.

### 2. Claim Risk Prediction

The dataset is divided into training and validation sets. SMOTE is applied to the training set to address class imbalance.

Three models — CART, XGBoost, and Neural Network — generate claim probabilities that are combined into an ensemble prediction.

### 3. Risk Interpretation

SHAP analysis identifies the features contributing most strongly to the predicted claim probability.

### 4. Premium Optimization

NSGA-II optimizes the weights of phenology-specific indicators by balancing insurer and farmer objectives. These weights contribute to the Viability Score and subsequent premium adjustment.

---

## Example Premium Calculation

The research demonstrates how the framework can dynamically modify premiums based on the estimated risk profile of an orchard.

For one example orchard:

| Metric                    |             Value |
| ------------------------- | ----------------: |
| Baseline Premium          |  $1,299.58 / acre |
| Initial Dynamic Premium   |     $55.00 / acre |
| Dormancy Adjustment       |              +10% |
| Bloom Adjustment          |              +20% |
| Maturation Adjustment     |              −10% |
| **Final Dynamic Premium** | **$65.34 / acre** |

This example illustrates the intended mechanism of moving from static actuarial pricing toward a more responsive risk-based premium framework.

---

## Limitations

The current research implementation has several limitations:

* The insurance policy dataset is partially simulated.
* The feature engineering is specialized for California almond phenology.
* SMOTE introduces synthetic minority-class observations.
* Ensemble modelling and multi-objective optimization increase computational complexity.
* Direct generalization to other crops or geographies requires additional validation.
* Real-world deployment would require further assessment of regulatory, fairness, and explainability requirements.

---

## Future Scope

Potential extensions include:

* 🌾 Extending the framework to additional crops
* 🛰️ Integrating satellite and UAV imagery
* 🌦️ Incorporating real-time weather data
* 📍 Expanding to additional agricultural regions
* 🔄 Developing a fully interactive intra-seasonal premium adjustment system
* ⚖️ Conducting formal fairness audits
* 🏛️ Evaluating regulatory and actuarial compliance
* ⚡ Improving computational efficiency for enterprise-scale deployment

---

## Research

**P.R.I.M.A.: Dynamic Premium Adjustment Model for Crop Insurance**

The research proposes an interpretable dynamic premium adjustment framework integrating:

**Phenology-Based Risk Features → Machine Learning → SHAP Explainability → NSGA-II Optimization → Viability Score → Dynamic Premium**

The research focuses on developing a transparent pricing mechanism that can adapt to changing agricultural risk while balancing insurer sustainability and farmer fairness.

---

## Author

**Aastha Chhabra**
Integrated MSc — Quantitative Economics & Data Science
Birla Institute of Technology, Mesra

**Research collaboration:**
Manish Kumar Pandey
Digital Innovation Lab, CQEDS
Birla Institute of Technology, Mesra

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@article{chhabra2026prima,
  title={P.R.I.M.A.: Dynamic Premium Adjustment Model for Crop Insurance},
  author={Chhabra, Aastha and Pandey, Manish Kumar},
  year={2026},
  institution={Birla Institute of Technology, Mesra}
}
```

---

## License

This repository is intended for research and educational purposes.

Please refer to the repository license for terms governing reuse and distribution.
