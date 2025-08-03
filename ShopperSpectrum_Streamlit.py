import streamlit as st
import pandas as pd
import numpy as np
import joblib

## Load the Model and Similarity Matrix ##
similarity_df = pd.read_csv(r"C:\Users\Nithya\product_similarity_matrix.csv", index_col=0)
kmeans = joblib.load(r"C:\Users\Nithya\kmeans.model")
scaler = joblib.load(r"C:\Users\Nithya\standard_scaler.pkl")

## Customer Segment Prediction ##
def predict_segment(recency, frequency, monetary):
    customer_data = pd.DataFrame([[recency, frequency, monetary]],columns=["Recency", "Frequency", "Monetary"])
    ## Applying Scaler before predicting values ##
    customer_data_scaled = scaler.transform(customer_data)
    cluster_label = kmeans.predict(customer_data_scaled)[0]
    
    segment_map = {
        0: "Occasional",
        1: "At-Risk",
        2: "High-Value",
        3: "Regular"
    }
    segment_name = segment_map.get(cluster_label, f"Cluster {cluster_label}")
    return cluster_label, segment_name

## Product Recommendation function ##
def recommend_items(input_item, similarity_df, top_n=5):
    input_item = input_item.strip().upper()
    if input_item not in similarity_df.columns:
        return f"⚠️ Product '{input_item}' not found. Please check the spelling or try another item."
    similar_items = similarity_df[input_item].drop(input_item)
    recommended = similar_items.sort_values(ascending=False).head(top_n)
    return recommended.index.tolist()

## Streamlit UI ##
st.set_page_config(page_title="Customer & Product Insights", layout="centered")
st.title("🧠 Customer Segmentation & 🎯 Product Recommendation")

## Sidebar navigation ##
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Clustering", "Recommendation"])

if page == "Home":
    st.title("🏠 Welcome")
    st.markdown("Use the sidebar to navigate to Clustering or Product Recommendations.")
    
elif page == "Clustering":
    st.title("📊 Customer Segmentation")
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Segment"):
        label, name = predict_segment(recency, frequency, monetary)
        st.subheader("🧬 Prediction Result")
        st.success(f"This customer belongs to: **{name}** (Cluster {label})")

elif page == "Recommendation":
    st.header("🔎 Product Recommendation")

    input_item = st.text_input("Enter Product Name")
    if st.button("Recommend"):
        output = recommend_items(input_item, similarity_df)
        if isinstance(output, str):
            st.warning(output)
        else:
            st.subheader("💡 Recommended Products:")
            for item in output:
                st.write(f"- {item}")
