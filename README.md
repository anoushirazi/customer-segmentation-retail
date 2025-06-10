#  Customer Segmentation for Retail

Customer segmentation using unsupervised machine learning and PCA to drive targeted marketing strategy in retail. This project demonstrates how to use transactional data to profile customer behavior and inform personalized campaigns.

---

## 📘 Project Overview

The goal of this project is to build a complete customer segmentation pipeline using retail transaction data. The approach combines unsupervised learning techniques like **KMeans**, **DBSCAN**, and **GMM** with **PCA** for dimensionality reduction. Each segment is profiled and aligned with actionable marketing strategies.

---

## 📈 Key Highlights

- Aggregated raw transaction-level data to the customer level.
- Engineered features: total spend, average spend, transaction count.
- Scaled data and encoded categorical variables.
- Performed cluster analysis using:
  - **KMeans + PCA** (best-performing)
  - **DBSCAN**
  - **Gaussian Mixture Model (GMM)**
- Determined optimal number of clusters using **Elbow** and **Silhouette** methods.
- Created customer profiles and assigned marketing strategies.
- Saved models using `joblib` for reuse and deployment.

---

## 🧭 Objectives

- Segment customers based on purchasing behavior and demographics.
- Enable personalized marketing strategies.
- Create interpretable, business-relevant clusters.
- Apply multiple clustering algorithms and compare performance.

---

## 🔧 Techniques Used

- **Data Cleaning & Aggregation** (customer-level transformation)
- **Feature Engineering**: Total/Avg Spend, Transaction Count
- **Encoding**: Gender → Numeric
- **Standardization**: `StandardScaler`
- **Clustering Algorithms**: KMeans, DBSCAN, GMM
- **Dimensionality Reduction**: PCA
- **Model Evaluation**: Silhouette Score, Elbow Method
- **Segment Profiling**: Cluster analysis + marketing strategy
- **Model Saving**: `joblib.dump()`

---

## 📊 Clustering Results

| Cluster | Segment Description              | Suggested Action                 |
|---------|----------------------------------|----------------------------------|
| 0       | Young, high-spending             | Loyalty program                  |
| 1       | Older, low-spending              | Reactivation email campaign      |
| 2       | Frequent, low-value transactions | Bundle/upsell offers             |
| 3       | High transaction count & spending| VIP program or priority tier     |

**🏆 Best Performing Model**: KMeans + PCA  
**Silhouette Score**: ~0.50–0.60 (most stable and interpretable)

---

## 📈 Visual Insights

- Elbow and Silhouette plots to identify optimal cluster count.
- PCA scatter plots colored by cluster labels.
- Spending distributions by gender and cluster.

📁 All visual outputs are stored in: `visuals/plots/`

---

## 📂 Project Structure

```bash
customer-segmentation-retail/
├── README.md                     # Project overview and documentation
├── requirements.txt              # Python dependencies
├── data/
│   └── raw/                      # Raw input dataset (excluded from Git)
├── notebooks/
│   └── customer_segmentation.ipynb  # Full modeling and analysis
├── src/                          # Source code
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
│   └── plots/                    # PCA, Elbow, and cluster plots
└── .gitignore
