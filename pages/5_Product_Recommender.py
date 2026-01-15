
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Product Recommender",
    page_icon="🎯",
    layout="wide"
)

# --- Data Loading and Feature Engineering ---
@st.cache_data
def load_and_prepare_data():
    """Loads data and engineers features for the recommendation model."""
    df = pd.read_csv("amazon.csv")

    # Basic cleaning from other pages
    for col in ['discounted_price', 'actual_price']:
        df[col] = df[col].str.replace('₹', '').str.replace(',', '').astype(float)
    df['rating'] = df['rating'].str.replace('|', '3.9').astype(float)
    df.dropna(subset=['product_name', 'category'], inplace=True)
    df = df.drop_duplicates(subset="product_id").reset_index(drop=True)

    # Feature Engineering from notebook
    df["full_text"] = (df["product_name"].astype(str) + " " +
                       df["about_product"].astype(str) + " " +
                       df["category"].astype(str)).str.lower()
    df["main_category"] = df["category"].apply(lambda x: str(x).split("|")[0].lower())

    def extract_num(text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else np.nan # Use np.nan for missing numeric values

    def extract_features(row):
        text = row["full_text"]
        features = {}
        features["watts"] = extract_num(text, r"(\d+)\s*w")
        features["typec"] = int(("type-c" in text) or ("type c" in text) or ("usb-c" in text))
        features["fast_charge"] = int("fast charge" in text or "quick charge" in text)
        features["braided"] = int("braided" in text)
        colors = ["black", "white", "red", "blue", "green"]
        for c in colors:
            features[f"color_{c}"] = int(c in text)
        return features

    feature_df = df.apply(extract_features, axis=1, result_type='expand')
    
    numeric_cols_for_scaling = [
        "watts", "typec", "fast_charge", "braided",
        "color_black", "color_white", "color_red", "color_blue", "color_green"
    ]
    
    # Ensure all these columns exist in feature_df, filling with NaN if not
    for col in numeric_cols_for_scaling:
        if col not in feature_df.columns:
            feature_df[col] = np.nan

    df = pd.concat([df, feature_df], axis=1)
    
    # Fill NaNs with 0 and then explicitly convert to float type
    for col in numeric_cols_for_scaling:
        df[col] = df[col].fillna(0).astype(float)

    return df, numeric_cols_for_scaling

# --- Recommendation Model ---
@st.cache_resource
def build_similarity_matrices(_df, _numeric_features):
    """Builds TF-IDF and numeric similarity matrices."""
    # TF-IDF for text similarity
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    text_matrix = tfidf.fit_transform(_df["full_text"])
    text_sim_matrix = cosine_similarity(text_matrix)

    # Scaled numeric features for feature similarity
    scaler = MinMaxScaler()
    numeric_scaled = scaler.fit_transform(_df[_numeric_features])
    numeric_sim_matrix = cosine_similarity(numeric_scaled)
    
    return text_sim_matrix, numeric_sim_matrix

def get_similarity_score(idx1, idx2, text_sim, num_sim, df):
    """Calculates a weighted similarity score."""
    text_score = text_sim[idx1, idx2]
    num_score = num_sim[idx1, idx2]
    
    # Boost score if products are in the same main category
    cat_boost = 1.2 if df.loc[idx1, "main_category"] == df.loc[idx2, "main_category"] else 1.0
    
    # Final weighted score
    return (0.6 * text_score + 0.4 * num_score) * cat_boost

def recommend_products(df, product_name, text_sim, num_sim, top_n=5):
    """Recommends top N products based on similarity."""
    if product_name not in df["product_name"].values:
        return None
    
    idx = df.index[df["product_name"] == product_name][0]
    
    scores = [(i, get_similarity_score(idx, i, text_sim, num_sim, df)) for i in range(len(df)) if i != idx]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in scores[:top_n]]
    
    return df.loc[top_indices, ["product_name", "main_category", "discounted_price", "rating"]]

# --- Main Page ---
def recommender_page():
    st.title("🎯 Product Recommendation Engine")
    st.markdown("""
    Select a product from the dropdown below to see a list of the top 5 most similar products.
    The recommendation is based on a combination of product description (text similarity) and key features (feature similarity).
    """)

    try:
        data, numeric_features = load_and_prepare_data()
        text_sim_matrix, numeric_sim_matrix = build_similarity_matrices(data, numeric_features)
    except Exception as e:
        st.error(f"Failed to build the recommendation model: {e}")
        st.stop()

    # --- Interactive Widget ---
    product_list = data['product_name'].unique()
    selected_product = st.selectbox(
        "Choose a Product to Get Recommendations For:",
        product_list,
        index=0
    )

    if st.button("Find Similar Products"):
        with st.spinner("Finding recommendations..."):
            recommendations = recommend_products(data, selected_product, text_sim_matrix, numeric_sim_matrix)
            
            st.subheader(f"Top 5 Recommendations for '{selected_product}':")
            
            if recommendations is not None and not recommendations.empty:
                st.table(recommendations.style.format({
                    "discounted_price": "₹{:.2f}",
                    "rating": "{:.1f} ⭐"
                }))
                st.markdown("""
                The table above presents the top 5 products most similar to your selected item. 
                These recommendations are generated by analyzing both textual descriptions and key product features, 
                aiming to suggest items that align with your interests.
                """)
            else:
                st.warning("Could not find any recommendations for this product.")

if __name__ == "__main__":
    recommender_page()
