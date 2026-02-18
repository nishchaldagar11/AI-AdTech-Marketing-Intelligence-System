
# 🚀 AdTech-Customer-Intelligence-System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=flat-square&logo=scikit-learn)

> An end-to-end applied Machine Learning project in the AdTech domain designed to optimize digital advertising strategy through customer segmentation, churn prediction, conversion modeling, and ROI-driven budget allocation.

---

## 📌 Project Overview

This project simulates a real-world AdTech marketing intelligence system that leverages Machine Learning to enhance campaign performance and support data-driven decision-making.

It integrates:

- 🔍 **Customer Segmentation** — KMeans Clustering  
- 📉 **Churn Prediction** — Random Forest Classifier  
- 🎯 **Conversion Prediction** — Random Forest  
- 💰 **ROI-Based Budget Optimization** — Multi-channel allocation  
- 🤖 **AI-Driven Strategic Recommendations**  
- 🌐 **REST API Integration Example**

---

## 🎯 Business Objective

Modern digital marketing platforms must:

- Identify high-value customers  
- Predict churn risk proactively  
- Improve conversion rates across channels  
- Optimize advertising budget allocation for maximum ROI  

This system demonstrates how AI-powered analytics can improve marketing efficiency and deliver measurable business impact.

---

## 🧠 Machine Learning Architecture

### 1️⃣ Customer Segmentation (Unsupervised Learning)

- **Algorithm:** KMeans Clustering  
- **Features:** Annual Income & Spending Score  
- **Output — Business-Ready Segments:**
  - 🏆 High Value Customers  
  - 💼 Budget Shoppers  
  - 👑 Premium Loyal Customers  
  - 😴 Low Engagement Customers  
  - ⚡ Young High Spenders  

---

### 2️⃣ Churn Prediction (Supervised Learning)

- **Model:** Random Forest Classifier  
- **Objective:** Predict likelihood of customer churn  
- **Evaluation Metrics:**
  - Classification Report  
  - ROC-AUC Score  
  - Confusion Matrix  

---

### 3️⃣ Conversion Prediction

- **Model:** Random Forest  
- **Objective:** Predict customer conversion probability  
- **Metric:** Accuracy Score  

---

## 💰 Multi-Channel Budget Optimization

Simulated advertising channels and intelligent budget reallocation:

| Channel | Optimization Metric |
|---------|--------------------|
| Google Ads | ROI × Conversion Rate |
| Meta Ads | ROI × Conversion Rate |
| LinkedIn Ads | ROI × Conversion Rate |

**Budget Reallocation Formula:**

```

Channel Score    = ROI × Conversion Rate
Optimized Budget = (Channel Score / Total Score) × Total Budget

````

This reflects multi-factor optimization logic used in real-world AdTech ecosystems.

---

## 🤖 AI-Driven Insights

The system generates strategic recommendations such as:

- Target premium campaigns toward high-value segments  
- Retarget low-engagement customers with personalized offers  
- Allocate higher spend to high-ROI channels  
- Use predictive insights to reduce churn risk  

---

## 🛠 Tech Stack

| Library | Purpose |
|----------|----------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | ML modeling & evaluation |
| `matplotlib` | Visualization |
| `seaborn` | Statistical plots |
| `requests` | REST API integration |

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nishchaldagar11/AdTech-Customer-Intelligence-System.git
cd AdTech-Customer-Intelligence-System
pip install pandas numpy seaborn matplotlib scikit-learn requests
````

---

## 🚀 Usage

Ensure `Mall_Customers.csv` is present in the project root directory, then run:

```bash
python adtech_project.py
```

**Expected Outputs:**

* 📊 Segmentation visualization plots
* 📈 Model performance metrics (ROC-AUC, accuracy, confusion matrix)
* 💰 Budget optimization chart
* 📁 Final enriched dataset saved as `AdTech_Final_Output.csv`

---

## 🌐 API Integration

Includes a REST API usage example via Python’s `requests` library to demonstrate how production-grade AdTech systems consume external data feeds within analytics pipelines.

---

## 📈 Business Impact

This project demonstrates how AI can:

* Improve customer targeting precision
* Reduce churn through proactive intervention
* Increase campaign efficiency across channels
* Optimize marketing spend allocation
* Enable scalable, data-driven growth strategies

---

## 📄 License

This project is licensed under the MIT License.

---


