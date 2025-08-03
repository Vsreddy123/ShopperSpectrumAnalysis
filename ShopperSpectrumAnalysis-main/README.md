# ShopperSpectrumAnalysis

Shopper Spectrum: A smart tool that helps online stores understand different types of customers and suggest products they’re likely to prefer—using machine learning and interactive visuals.

**Overview**

Shopper Spectrum is a data-driven solution designed to enhance e-commerce platforms by uncovering customer insights and providing personalized product recommendations. Using transaction data, this project performs customer segmentation through RFM analysis and develops a collaborative filtering-based recommendation system. The goal is to enable targeted marketing, improve customer experience, and optimize business strategies.

**Problem Statement**

The e-commerce industry generates vast amounts of transaction data daily, which holds valuable insights into customer purchasing behavior. Analyzing this data allows businesses to:

*Identify meaningful customer segments

*Deliver relevant product recommendations

*Improve customer retention

*Optimize inventory and pricing strategies

This project focuses on analyzing online retail transactions to uncover buying patterns, segment customers based on Recency, Frequency, and Monetary (RFM) metrics, and build a product recommendation system using collaborative filtering techniques.

**Use Cases**

-->Customer Segmentation for targeted marketing campaigns

-->Personalized Product Recommendations to boost sales and improve user experience

-->Identifying At-Risk Customers for retention efforts

-->Dynamic Pricing Strategies based on purchase behavior

-->Inventory Management driven by customer demand patterns

**Technology & Methods**

Unsupervised Machine Learning for customer segmentation via clustering (K-means)

Collaborative Filtering for generating personalized product recommendations

Data preprocessing and scaling using StandardScaler

Model persistence with kmeans.model

**Files & Components**

*ShopperSpectrumAnalysis_USVL.ipynb

Jupyter Notebook containing full exploratory data analysis, RFM segmentation, and recommendation system implementation.

*ShopperSpectrum_Streamlit.py

Python script with core functions for Streamlit interface design , clustering, and generating recommendations.

*kmeans.model

Saved K-means clustering model used for customer segmentation.

*standard_scaler.pkl

Pickle file containing the scaler used for feature normalization.

**Note:** Replace placeholder URLs and user-specific paths as necessary.
