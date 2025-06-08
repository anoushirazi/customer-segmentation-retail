### About
Customer segmentation using clustering and PCA for targeted retail marketing. Unsupervised learning meets business strategy.

# Customer Segmentation for Retail

Identify and understand key customer segments using unsupervised machine learning. This project uses retail transaction data to cluster customers into meaningful groups, enabling targeted marketing strategies.

---

## 📌 Project Summary

In this project, I developed an end-to-end customer segmentation pipeline using clustering algorithms like KMeans, DBSCAN, and GMM. The analysis is enriched with dimensionality reduction (PCA), behavioral profiling, and actionable marketing recommendations.

This work combines data science, machine learning, and business analytics to deliver real-world value.

---

## 🧭 Objectives

- Group customers based on purchasing behavior and demographics
- Enable data-driven, targeted marketing campaigns
- Visualize and profile segments to guide business strategy

---

## 📂 Project Structure

```bash
customer-segmentation-retail/
├── README.md
├── requirements.txt
├── data/
│   └── raw/                      # Raw dataset (not committed)
├── notebooks/
│   └── customer_segmentation.ipynb  # Full EDA & modeling notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── clustering_models.py
│   ├── evaluation.py
│   ├── main.py
│   └── utils.py
├── models/
│   ├── model.joblib              # Trained KMeans model
│   └── scaler.joblib             # Fitted StandardScaler
├── visuals/
│   └── plots/                    # Elbow curves, PCA cluster plots, etc.
└── .gitignore
