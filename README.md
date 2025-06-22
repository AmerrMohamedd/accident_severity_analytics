# Accident Severity Analysis

Analyzing over 7 million US traffic accident records (2016–2023) to uncover patterns, predict severity, and deliver actionable insights through dashboards and machine learning.

---

## 1. Executive Summary

**Problem:**  
Traffic accidents cause major societal and economic losses in the US. Understanding when, where, and why severe accidents happen can help save lives.

**Solution:**  
We analyzed 7M+ accident records using exploratory data analysis (EDA), built a machine learning classification model to predict severity, and created an interactive dashboard for stakeholders to explore insights.

**Impact:**  
- Identified high-risk locations and peak times.
- Built a predictive model with promising performance.
- Delivered visual insights that can guide safety policies and emergency responses.

---

## 2. Business Problem & Objective

- **Why this matters:** Improving road safety and emergency response efficiency.
- **Who benefits:** Traffic management authorities, city planners, emergency responders.
- **Goal:** Help stakeholders anticipate accident severity based on real-time data.

---

## 3. Data & Methodology

- **Dataset:** US traffic accidents (2016–2023), ~7.2M records.
- **Process:** Followed the CRISP-DM process — from data cleaning, EDA, feature engineering, to model training and dashboard deployment.
- **ML Pipeline (FTI):**
  - **Features:** Time, location, weather, and road conditions.
  - **Training:** Used classification algorithms (Logistic Regression, Random Forest, XGBoost).
  - **Inference:** Severity level prediction.
- **Tools & Tech:** Python, Pandas, scikit-learn, Plotly Dash, Jupyter, Git.

---

## 4. Results & Insights

- Found that accidents are most severe at night and during adverse weather.
- Certain US states and intersections are high-risk clusters.
- Predictive model achieved strong accuracy in classifying severity levels.
- Visual dashboard enables filtering by time, region, and cause.

---

## 5. Conclusion & Next Steps

- **Takeaways:** Data science can uncover patterns that improve public safety.
- **Next steps:** Enhance real-time prediction and integrate live traffic data.
- **Limitations:** Some records had missing or imbalanced data; further cleaning and augmentation could improve results.

---

## 6. Technical Setup

To run the project locally:

```bash
git clone https://github.com/AmerrMohamedd/accident_severity_analysis.git
cd accident_severity_analysis
pip install -r requirements.txt
jupyter notebook
