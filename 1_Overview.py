import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Amazon Sales Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_and_clean_data():
    """Loads, cleans, and preprocesses the Amazon sales data."""
    df = pd.read_csv("amazon.csv")

    # Clean and convert price columns
    for col in ['discounted_price', 'actual_price']:
        df[col] = df[col].str.replace('₹', '').str.replace(',', '').astype(float)

    # Clean and convert percentage
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)

    # Fix the single incorrect rating value
    df['rating'] = df['rating'].str.replace('|', '3.9').astype(float)

    # Clean and convert rating_count
    df['rating_count'] = df['rating_count'].str.replace(',', '', regex=False).astype(float)
    
    # Impute missing rating_count with the median
    median_rating_count = df['rating_count'].median()
    df['rating_count'] = df['rating_count'].fillna(median_rating_count)

    # Drop rows with no category, as they are not useful for analysis
    df.dropna(subset=['category'], inplace=True)
    
    # Extract main category
    df['main_category'] = df['category'].apply(lambda x: x.split('|')[0])

    return df

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🛒 Amazon Sales Dashboard")
    st.markdown("""
    Welcome to the Amazon Sales Analysis Dashboard. This interactive tool explores a dataset of over 1,000 Amazon products. 
    Use the sidebar to navigate through different deep-dive analyses.
    """)

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

    # --- Overview Section ---
    st.header("Dataset Overview")
    st.markdown("""
    This section provides a high-level overview of the dataset and key performance indicators.
    """)
    
    # Key Metrics
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    avg_rating = round(filtered_data['rating'].mean(), 2)
    avg_discount = int(filtered_data['discount_percentage'].mean())
    total_products = filtered_data.shape[0]
    total_categories = filtered_data['main_category'].nunique()

    col1.metric("Total Products Analyzed", f"{total_products:,}")
    col2.metric("Average Rating", f"{avg_rating} ⭐")
    col3.metric("Average Discount", f"{avg_discount}%")
    col4.metric("Unique Categories", f"{total_categories}")

    # --- Visualizations ---
    st.header("Initial Visual Insights")
    
    # Distribution of Ratings
    st.subheader("Distribution of Product Ratings")
    fig_rating = px.histogram(filtered_data, x="rating", nbins=20, title="Product Rating Distribution",
                              labels={'rating': 'Rating'}, color_discrete_sequence=['#636EFA'])
    fig_rating.update_layout(bargap=0.1)
    st.plotly_chart(fig_rating, use_container_width=True)
    
    st.markdown("""
    The histogram above shows that most products on Amazon have high ratings, primarily clustered between 4.0 and 4.5 stars. 
    This indicates a general satisfaction among customers for the products in this dataset.
    """)

    # Distribution of Discount Percentage
    st.subheader("Distribution of Discounts")
    fig_discount = px.histogram(filtered_data, x="discount_percentage", nbins=20, title="Discount Percentage Distribution",
                                labels={'discount_percentage': 'Discount (%)'}, color_discrete_sequence=['#00CC96'])
    fig_discount.update_layout(bargap=0.1)
    st.plotly_chart(fig_discount, use_container_width=True)
    
    st.markdown("""
    This histogram illustrates that a significant portion of products offer discounts between 50% and 60%. 
    This suggests that Amazon frequently provides substantial discounts, making products more accessible to a wider audience.
    """)

    # Most Frequent Categories
    st.subheader("Top 10 Most Frequent Product Categories")
    category_counts = filtered_data['main_category'].value_counts().nlargest(10).reset_index()
    category_counts.columns = ['category', 'count']
    
    fig_cat = px.bar(category_counts, y='category', x='count', orientation='h',
                     title="Top 10 Product Categories",
                     labels={'count': 'Number of Products', 'category': 'Category'},
                     color='count', color_continuous_scale=px.colors.sequential.Viridis)
    fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cat, use_container_width=True)
    
    st.markdown("""
    The bar chart highlights the top 10 most frequent product categories in the dataset. 
    Understanding these dominant categories can inform inventory management and marketing strategies.
    """)


if __name__ == "__main__":
    main()
