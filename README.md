# Absenteeism at Work: Predictive Analytics & Power BI Dashboard

An end-to-end HR analytics project combining Python machine learning with a four-page interactive Power BI dashboard to analyze, predict, and visualize employee absenteeism patterns. The Random Forest model achieved 84% classification accuracy, identifying BMI, reason for absence, and transportation expense as the strongest predictors of absenteeism risk.

<p align="center">
  <a href="https://app.powerbi.com/groups/me/reports/a3524fb6-4abb-4bad-b58a-c51b6ccbe14c/38db5aa2962648d1b160be?experience=power-bi">View Live Power BI Dashboard</a> · <a href="Dataset 4 The Absenteeism Dataset.pdf">View Project Report</a>
</p>

<p align="center">
  <img src="images/dashboard 1 .png" width="680" alt="Absenteeism at Work Power BI Dashboard - Overview Page"/>
</p>

<p align="center"><em>Dashboard Page 1 - Absenteeism trends overview showing patterns by month, season, reason for absence, and education level.</em></p>

---

## Overview

Employee absenteeism is one of the most measurable and underanalyzed costs in workforce management. Beyond the direct cost of lost productivity, unplanned absences disrupt scheduling, increase overtime, and create cascading operational problems. Understanding which employees are at highest risk of excessive absenteeism - and why - allows HR teams to intervene proactively rather than react after absences occur.

This project builds that understanding from two directions. The Python pipeline cleans and encodes a real-world absenteeism dataset, performs exploratory data analysis to surface seasonal and demographic patterns, and trains multiple classification models to predict absenteeism risk. The Power BI dashboard then makes those findings accessible to HR managers and business leaders who need actionable insights without reading a model summary.

The combination of a predictive ML pipeline and a business-facing dashboard is exactly the workflow used in production HR analytics systems at large organizations - this project demonstrates that full workflow end to end.

---

## The Dataset

| | |
|---|---|
| Source | Absenteeism at Work Dataset |
| Format | Excel / CSV |
| Key Variables | Reason for absence, month, season, transportation expense, distance from residence, service time, age, BMI, education level, social drinking, social smoking, absenteeism hours |
| Target Variable | Absenteeism level (High / Moderate / Low) |
| Tools | Python, Jupyter Notebook, Power BI Desktop |

---

## Python ML Pipeline

### Stage 1 - Data Cleaning and Preprocessing

The raw dataset uses numeric codes for categorical variables - seasons are coded 1-4, education levels 1-4, and reasons for absence 1-28. Leaving these as raw numbers would cause the model to treat them as continuous variables with ordinal meaning they do not have. The first preprocessing step was mapping all coded variables to readable labels.

```python
# Example - mapping reason for absence codes to readable categories
reason_map = {
    1: 'Infectious Diseases', 2: 'Neoplasms', 3: 'Blood Diseases',
    13: 'Musculoskeletal', 14: 'Congenital Conditions', 19: 'Injury',
    23: 'Medical Consultation', 28: 'Dental Consultation'
    # ... full mapping in notebook
}
df['Reason for Absence'] = df['Reason for Absence'].map(reason_map)
```

Season labels (Spring, Summer, Autumn, Winter), education levels (High School, Graduate, Post-Graduate, Master/Doctor), and all other coded columns were mapped before any analysis was performed. This is a critical step - visualizations and model outputs built on unmapped codes are uninterpretable to business stakeholders.

---

### Stage 2 - Exploratory Data Analysis

EDA examined how absenteeism hours varied across months, seasons, education levels, and reasons for absence to identify patterns before any predictive modeling.

<p align="center">
  <img src="images/distribution of absenteeism hours .png" width="600" alt="Distribution of absenteeism hours across the dataset"/>
</p>

<p align="center"><em>Distribution of absenteeism hours - right-skewed with most employees showing low hours but a long tail of high-absence outliers who drive disproportionate operational impact.</em></p>

**Seasonal patterns.** Absenteeism is highest in Winter, consistent with increased illness rates and weather-related disruptions. Spring shows the second highest rates, while Summer shows the lowest - a pattern that has direct implications for seasonal staffing strategies.

**Monthly patterns.** March shows the highest absenteeism volume across the dataset. Planning additional staffing coverage or implementing wellness interventions in February and March would directly address the peak period.

<p align="center">
  <img src="images/top reasons for absence .png" width="600" alt="Top reasons for employee absence ranked by frequency"/>
</p>

<p align="center"><em>Top reasons for absence ranked by frequency - medical consultations and musculoskeletal conditions dominate, pointing toward preventive health programs as a high-ROI intervention.</em></p>

**Top reasons for absence.** Medical consultations and musculoskeletal conditions account for the largest share of absences. These are both categories where employer-sponsored preventive health programs - on-site clinics, ergonomic assessments, physical therapy access - have documented ROI in reducing absenteeism costs.

<p align="center">
  <img src="images/absenteeism by BMI Category .png" width="580" alt="Absenteeism hours by BMI category"/>
</p>

<p align="center"><em>Absenteeism by BMI category - employees in the Obese category show measurably higher average absenteeism hours, making BMI one of the strongest health-lifestyle predictors in the dataset.</em></p>

<p align="center">
  <img src="images/absenteeism by Risky Lifestyle .png" width="580" alt="Absenteeism patterns by lifestyle risk factors including social drinking and smoking"/>
</p>

<p align="center"><em>Absenteeism by lifestyle risk factors - social drinking and smoking show a compounding effect on absence hours when combined with high BMI, surfacing a high-risk employee profile for targeted wellness intervention.</em></p>

---

### Stage 3 - Predictive Modeling

Multiple classification models were trained and compared to predict employee absenteeism level. The target variable was engineered into three classes - High, Moderate, and Low absenteeism - based on hours thresholds derived from the distribution analysis.

<p align="center">
  <img src="images/predection model .png" width="600" alt="Model comparison showing accuracy scores across all trained classifiers"/>
</p>

<p align="center"><em>Model performance comparison across all trained classifiers. Random Forest achieves the highest accuracy at 84%, outperforming Decision Tree and Logistic Regression baselines.</em></p>

**Random Forest was selected as the final model** based on its superior accuracy of 84% across the held-out test set. Random Forest outperforms a single Decision Tree by averaging predictions across many trees - reducing the variance that causes individual trees to overfit on small datasets. It also handles the mix of continuous and categorical features in this dataset without requiring separate preprocessing for each variable type.

<p align="center">
  <img src="images/Random forest confusion Matrix .png" width="560" alt="Random Forest confusion matrix showing classification breakdown by absenteeism level"/>
</p>

<p align="center"><em>Random Forest confusion matrix - breaking down correct and incorrect classifications across High, Moderate, and Low absenteeism classes. The model performs strongest on the Low absenteeism class and shows some overlap between Moderate and High - expected behavior given the natural ambiguity at class boundaries.</em></p>

**Feature importance findings:**

The three strongest predictors of absenteeism risk identified by the Random Forest model are:

**BMI** - the single strongest predictor. Employees in the Obese BMI category show significantly higher absenteeism hours on average, and the model weighted this variable most heavily across all splits. This finding directly supports the case for employer-sponsored wellness programs targeting weight management and preventive health.

**Reason for Absence** - the second strongest predictor. Medical and musculoskeletal reasons produce consistently higher absenteeism hours than administrative or personal reasons. Knowing the reason category for an employee's first absence in a period predicts whether subsequent absences are likely.

**Transportation Expense** - the third strongest predictor. Higher transportation costs correlate with higher absenteeism, likely because employees with long or expensive commutes face more logistical barriers to attendance. This finding supports flexible work arrangements as an absenteeism reduction strategy for high-commute employees.

---

## Four-Page Power BI Dashboard

The dashboard was designed for HR managers and business leaders who need to act on absenteeism data - not data scientists who need to evaluate model performance. Each page answers a specific operational question.

---

### Page 1 - Absenteeism Trends Overview

<p align="center">
  <img src="images/dashboard 1 .png" width="640" alt="Dashboard Page 1 - Absenteeism Overview"/>
</p>

<p align="center"><em>Overview page - absenteeism patterns by month, season, reason for absence, and education level. The starting point for any HR review of workforce attendance trends.</em></p>

This page answers: **When does absenteeism peak and why?**

It shows monthly and seasonal absenteeism trends, reason for absence breakdowns, and education level comparisons. An HR manager reviewing this page immediately sees that March is the highest-risk month and Winter the highest-risk season, and can plan staffing coverage accordingly.

---

### Page 2 - Health and Lifestyle Analysis

<p align="center">
  <img src="images/dashboard 2 .png" width="640" alt="Dashboard Page 2 - Health and Lifestyle Analysis"/>
</p>

<p align="center"><em>Health and lifestyle page - BMI category, social drinking, social smoking, and their combined effect on absenteeism hours. Designed to support wellness program targeting decisions.</em></p>

This page answers: **Which health and lifestyle factors drive the highest absenteeism?**

It surfaces the BMI and lifestyle risk findings from EDA in an interactive format. An HR director reviewing this page can identify the specific employee profiles - high BMI, social smoking, social drinking - that carry the highest absenteeism risk and design targeted wellness interventions accordingly.

---

### Page 3 - Workplace Factor Analysis

<p align="center">
  <img src="images/dashboard 3 .png" width="640" alt="Dashboard Page 3 - Workplace Factor Analysis"/>
</p>

<p align="center"><em>Workplace factors page - transportation expense, distance from residence, service time, and workload metrics correlated with absenteeism. Supports operational and HR policy decisions.</em></p>

This page answers: **Which workplace and logistical factors predict higher absenteeism?**

It visualizes transportation expense, commute distance, service time, and workload variables against absenteeism hours. The transportation expense finding - one of the top three model predictors - is made actionable here by showing which expense brackets carry the highest risk, allowing HR to target flexible work or commuter benefit programs at the right employee segments.

---

### Page 4 - KPI Summary

<p align="center">
  <img src="images/dashboard 4 .png" width="640" alt="Dashboard Page 4 - KPI Summary with card visuals"/>
</p>

<p align="center"><em>KPI summary page - card visuals showing total absenteeism hours, average hours per employee, highest-risk month, most common reason for absence, and model accuracy. Designed for executive review.</em></p>

This page answers: **What are the headline numbers an executive needs to see?**

Card visuals display total absenteeism hours, average hours per employee, the highest-risk month, the most common reason for absence, and the predictive model accuracy. This is the page a CEO or CHRO would open first - it provides the summary metrics needed for a board presentation or budget discussion without requiring any navigation through the analytical pages.

---

## Key Findings and Business Recommendations

**Finding 1: Winter and March are the highest-risk periods.**
Recommendation: Pre-approve temporary staffing contracts in January for deployment in February-March. Implement a targeted wellness campaign in January focused on flu prevention and mental health support before the peak period begins.

**Finding 2: BMI is the strongest individual predictor of absenteeism risk.**
Recommendation: Employer-sponsored wellness programs targeting weight management, nutrition counseling, and physical activity have a documented ROI in absenteeism reduction. The model provides a quantitative basis for justifying this investment to finance teams.

**Finding 3: Medical consultations and musculoskeletal conditions dominate absence reasons.**
Recommendation: On-site or near-site medical clinic access and ergonomic workplace assessments directly address the two largest absence categories. These interventions are measurable - before and after absenteeism rates can be tracked in the Power BI dashboard.

**Finding 4: Transportation expense predicts absenteeism.**
Recommendation: Employees with the highest transportation costs should be prioritized for flexible work arrangements or commuter benefit programs. The scatter plot on Page 3 identifies the expense threshold above which absenteeism risk increases significantly.

---

## Repository Structure

```
Absenteeism-At-Work-PowerBI-ML/
│
├── images/
│   ├── dashboard 1 .png                          # Page 1 - Absenteeism Trends Overview
│   ├── dashboard 2 .png                          # Page 2 - Health and Lifestyle Analysis
│   ├── dashboard 3 .png                          # Page 3 - Workplace Factor Analysis
│   ├── dashboard 4 .png                          # Page 4 - KPI Summary
│   ├── Random forest confusion Matrix .png       # Model evaluation output
│   ├── predection model .png                     # Model comparison chart
│   ├── distribution of absenteeism hours .png    # EDA - hours distribution
│   ├── top reasons for absence .png              # EDA - absence reasons ranked
│   ├── absenteeism by BMI Category .png          # EDA - BMI vs absenteeism
│   └── absenteeism by Risky Lifestyle .png       # EDA - lifestyle risk factors
│
├── Absenteeism_Analysis.ipynb                    # Full Python pipeline - cleaning, EDA, modeling
├── Dataset 4 The Absenteeism Dataset.pdf         # Project report and documentation
├── absenteeism_data.csv                          # Source dataset
└── README.md
```

---

## Getting Started

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the notebook
jupyter notebook Absenteeism_Analysis.ipynb
```

To explore the Power BI dashboard interactively, visit the live link at the top of this README. All four pages are accessible without a Power BI account.

---

## Limitations and What I Would Do Next

The dataset covers a single organization over a defined time period. Generalizing the model's findings - particularly the BMI and transportation expense thresholds - to other organizations requires validation on broader workforce data.

The three-class target variable (High, Moderate, Low) introduces ambiguity at class boundaries. Some employees classified as Moderate would be borderline High, and the confusion matrix confirms this is where the model's 16% error rate is concentrated. A regression model predicting exact absenteeism hours rather than a categorical class would eliminate this boundary problem and provide more granular risk scores.

Integrating real-time HR data via a live Power BI data connection - rather than a static CSV - would convert this from a historical analysis into a live workforce monitoring tool that updates automatically as new attendance records are entered.

---

## Tools and Technologies

Python · Pandas · Scikit-Learn · Random Forest · Matplotlib · Seaborn · Jupyter Notebook · Microsoft Power BI

---

## Related Projects

- **[Pharmaceutical Sales Analytics - Power BI](https://github.com/TejashwiniSaravanan/Drug-Sales-Analysis-PowerBI)** - Star Schema BI solution for global drug sales and regulatory compliance
- **[Clinical Trial Patient Selection](https://github.com/TejashwiniSaravanan/Clinical-Trial-Patient-Selection-Optimization)** - Classification modeling for patient screening using Orange Data Mining
- **[Healthcare Analytics - PySpark ML & GCP Strategy](https://github.com/TejashwiniSaravanan/Healthcare-Analytics-PySpark-ML-GCP-Strategy)** - Cloud architecture for real-time patient monitoring

---

## About Me

**Tejashwini Saravanan** - Master's student in Data Analytics at Seattle Pacific University, focused on healthcare data engineering, HR analytics, and scalable ML pipelines.

[LinkedIn](https://www.linkedin.com/in/tejashwinisaravanan/) · [GitHub](https://github.com/TejashwiniSaravanan)

---

*Dataset: Absenteeism at Work · Tools: Python, Microsoft Power BI · Seattle Pacific University*
