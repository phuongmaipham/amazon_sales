
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import nltk

# --- Page Configuration ---
st.set_page_config(
    page_title="Product & Category Analysis",
    page_icon="📦",
    layout="wide"
)

# --- Download NLTK stopwords ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# --- Data Loading (from main page) ---
@st.cache_data
def load_and_clean_data():
    """Loads, cleans, and preprocesses the Amazon sales data."""
    df = pd.read_csv("amazon.csv")
    for col in ['discounted_price', 'actual_price']:
        df[col] = df[col].str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
    df['rating'] = df['rating'].str.replace('|', '3.9').astype(float)
    df['rating_count'] = df['rating_count'].str.replace(',', '', regex=False).astype(float)
    median_rating_count = df['rating_count'].median()
    df['rating_count'] = df['rating_count'].fillna(median_rating_count)
    df.dropna(subset=['category'], inplace=True)
    df['main_category'] = df['category'].apply(lambda x: x.split('|')[0])
    return df

# --- Helper Function for Keywords ---
def extract_keywords(text):
    if not isinstance(text, str):
        return []
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in tokens if word not in stop_words and len(word) > 2]

# --- Main Page ---
def product_analysis_page():
    """Creates the content for the product analysis page."""
    st.title("📦 Product and Category Deep Dive")
    st.markdown("This page analyzes product categories, popular product names, and common keywords found in product titles.")

    # Download stopwords
    download_nltk_data()

    # Load data
    try:
        data = load_and_clean_data()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Category Filter
    all_categories = data['main_category'].unique().tolist()
    selected_categories = st.sidebar.multiselect(
        "Select Category",
        options=all_categories,
        default=all_categories
    )

    # Rating Filter
    min_rating_val, max_rating_val = float(data['rating'].min()), float(data['rating'].max())
    rating_range = st.sidebar.slider(
        "Minimum Rating",
        min_value=min_rating_val,
        max_value=max_rating_val,
        value=min_rating_val,
        step=0.1
    )

    # Price Range Filter
    min_price_val, max_price_val = float(data['discounted_price'].min()), float(data['discounted_price'].max())
    price_range = st.sidebar.slider(
        "Discounted Price Range (₹)",
        min_value=min_price_val,
        max_value=max_price_val,
        value=(min_price_val, max_price_val),
        step=100.0
    )

    # Apply filters
    filtered_data = data[
        (data['main_category'].isin(selected_categories)) &
        (data['rating'] >= rating_range) &
        (data['discounted_price'] >= price_range[0]) &
        (data['discounted_price'] <= price_range[1])
    ]

    # Check if filtered_data is empty
    if filtered_data.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        st.stop() # Stop execution if no data

    # --- Category Analysis ---
    st.header("Analysis by Product Category")
    
    # Group by main category and calculate average metrics
    category_agg = filtered_data.groupby('main_category').agg(
        average_rating=('rating', 'mean'),
        average_discount=('discount_percentage', 'mean'),
        total_products=('product_id', 'count')
    ).reset_index()

    st.subheader("Product Count by Main Category")
    col_graph_1, col_data_1 = st.columns([2,1]) # Reversed order
    with col_graph_1:
        fig_cat_count = px.bar(category_agg.sort_values(by="total_products", ascending=True), 
                               y='main_category', x='total_products', orientation='h',
                               title="Number of Products per Main Category",
                               labels={'total_products': 'Number of Products', 'main_category': 'Main Category'},
                               color='total_products', color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_cat_count, use_container_width=True)
    with col_data_1:
        st.markdown("##### Category Metrics")
        st.dataframe(category_agg.sort_values(by="total_products", ascending=False))

    st.markdown("""
    This chart and table provide an overview of product distribution across different main categories, 
    along with their average ratings and discounts. It helps identify dominant product segments and their performance.
    """)

    # --- Popular Products ---
    st.header("Most Popular Products")
    st.markdown("Products are ranked here by their `rating_count`, which indicates popularity.")
    
    popular_products = filtered_data.sort_values(by="rating_count", ascending=False).head(10)
    col_graph_2, col_data_2 = st.columns([2,1]) # Reversed order
    with col_graph_2:
        fig_popular = px.bar(
            popular_products,
            x='rating_count',
            y='product_name',
            orientation='h',
            title="Top 10 Most Popular Products by Rating Count",
            labels={'rating_count': 'Number of Ratings', 'product_name': 'Product Name'},
            color='rating',
            color_continuous_scale=px.colors.sequential.Plasma,
            hover_data=['main_category', 'discounted_price']
        )
        fig_popular.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_popular, use_container_width=True)
    with col_data_2:
        st.markdown("##### Top 10 Popular Products (by Rating Count)")
        st.dataframe(popular_products[['product_name', 'main_category', 'rating', 'rating_count', 'discounted_price']])

    st.markdown("""
    The bar chart displays the top 10 products with the highest number of ratings, indicating their popularity. 
    This insight can be valuable for understanding market demand and identifying best-selling items.
    """)

    # --- Word Cloud ---
    st.header("Common Keywords in Product Names")
    st.markdown("The word cloud below highlights the most frequent terms used in product names, giving us a sense of common features and types.")

    # Generate keywords
    filtered_data['keywords'] = filtered_data['product_name'].apply(extract_keywords)
    all_keywords = [kw for sublist in filtered_data['keywords'] for kw in sublist]
    
    if all_keywords:
        text = ' '.join(all_keywords)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(text)
        
        col_graph_3, col_data_3 = st.columns([2,1]) # Reversed order
        with col_graph_3:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        with col_data_3:
            st.markdown("##### Top Product Name Keywords")
            keyword_counts = pd.Series(all_keywords).value_counts().head(10).reset_index()
            keyword_counts.columns = ['Keyword', 'Frequency']
            st.dataframe(keyword_counts)
            
        st.markdown("""
        The word cloud visually represents the most common words found in product names. 
        Larger words appear more frequently, indicating key product features or types that resonate with customers.
        """)
    else:
        st.warning("Could not generate keywords to create a word cloud.")


if __name__ == "__main__":
    product_analysis_page()
