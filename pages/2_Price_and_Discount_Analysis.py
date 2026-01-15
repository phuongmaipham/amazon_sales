
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Price & Discount Analysis",
    page_icon="💰",
    layout="wide"
)

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

# --- Main Page ---
def price_analysis_page():
    """Creates the content for the price analysis page."""
    st.title("💰 Price and Discount Deep Dive")
    st.markdown("""
    This section explores the relationships between product prices, discounts, and customer engagement. 
    How do discounts affect popularity? Is there a strong link between the original price and the sale price?
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

    # Discount Percentage Filter
    min_discount_val, max_discount_val = float(data['discount_percentage'].min()), float(data['discount_percentage'].max())
    discount_range = st.sidebar.slider(
        "Discount Percentage Range (%)",
        min_value=min_discount_val,
        max_value=max_discount_val,
        value=(min_discount_val, max_discount_val),
        step=1.0
    )

    # Apply filters
    filtered_data = data[
        (data['main_category'].isin(selected_categories)) &
        (data['rating'] >= rating_range) &
        (data['discounted_price'] >= price_range[0]) &
        (data['discounted_price'] <= price_range[1]) &
        (data['discount_percentage'] >= discount_range[0]) &
        (data['discount_percentage'] <= discount_range[1])
    ]

    # Check if filtered_data is empty
    if filtered_data.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        st.stop() # Stop execution if no data

    # --- Correlation Analysis ---
    st.header("Price Correlation Analysis")

    # Actual vs. Discounted Price
    st.subheader("Actual Price vs. Discounted Price")
    col_graph_1, col_data_1 = st.columns([2,1]) # Reversed order
    with col_graph_1:
        fig_price_corr = px.scatter(
            filtered_data, x="actual_price", y="discounted_price",
            opacity=0.6,
            trendline="ols",
            trendline_color_override="red",
            title="Correlation of Actual and Discounted Prices",
            labels={
                "actual_price": "Actual Price (₹)",
                "discounted_price": "Discounted Price (₹)"
            }
        )
        st.plotly_chart(fig_price_corr, use_container_width=True)
    with col_data_1:
        st.markdown("##### Descriptive Statistics")
        st.dataframe(filtered_data[['actual_price', 'discounted_price']].describe())
    
    price_corr = filtered_data['actual_price'].corr(filtered_data['discounted_price'])
    st.markdown(f"""
    The scatter plot above shows a strong positive correlation (coefficient: `{price_corr:.2f}`) between the actual price and the discounted price. 
    This indicates that more expensive products tend to have higher discounted prices, maintaining a consistent pricing structure even after discounts.
    """)

    # Discount vs. Popularity
    st.subheader("Discount % vs. Popularity (Rating Count)")
    col_graph_2, col_data_2 = st.columns([2,1]) # Reversed order
    with col_graph_2:
        fig_discount_pop = px.scatter(
            filtered_data, x="discount_percentage", y="rating_count",
            opacity=0.6,
            title="Does Higher Discount Mean More Ratings?",
            labels={
                "discount_percentage": "Discount Percentage (%)",
                "rating_count": "Number of Ratings"
            }
        )
        st.plotly_chart(fig_discount_pop, use_container_width=True)
    with col_data_2:
        st.markdown("##### Descriptive Statistics")
        st.dataframe(filtered_data[['discount_percentage', 'rating_count']].describe())
    
    discount_corr = filtered_data['discount_percentage'].corr(filtered_data['rating_count'])
    st.markdown(f"""
    This plot reveals a very weak correlation (coefficient: `{discount_corr:.2f}`) between discount percentage and the number of ratings. 
    This suggests that simply offering a higher discount does not necessarily lead to a significant increase in product popularity or customer reviews.
    """)

    # --- Price vs. Customer Rating ---
    st.header("Price vs. Customer Rating")
    st.markdown("Does a higher price tag mean a better rating? Let's investigate.")
    col_graph_3, col_data_3 = st.columns([2,1]) # Reversed order
    with col_graph_3:
        fig_price_rating = px.scatter(
            filtered_data, x="actual_price", y="rating",
            opacity=0.5,
            title="Actual Price vs. Product Rating",
            labels={
                "actual_price": "Actual Price (₹)",
                "rating": "Rating (out of 5)"
            },
            hover_data=['product_name']
        )
        fig_price_rating.update_traces(marker=dict(size=8,
                                                  line=dict(width=1,
                                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
        st.plotly_chart(fig_price_rating, use_container_width=True)
    with col_data_3:
        st.markdown("##### Descriptive Statistics")
        st.dataframe(filtered_data[['actual_price', 'rating']].describe())
    
    price_rating_corr = filtered_data['actual_price'].corr(filtered_data['rating'])
    st.markdown(f"""
    The correlation between actual price and product rating is very weak (coefficient: `{price_rating_corr:.2f}`). 
    This indicates that customers rate products more on their quality and experience rather than just their price point.
    """)


if __name__ == "__main__":
    price_analysis_page()
