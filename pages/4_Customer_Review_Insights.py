
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
    page_title="Customer Review Insights",
    page_icon="🗣️",
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
    # Add custom stopwords that are common in reviews but not informative
    custom_stopwords = ['product', 'good', 'quality', 'price', 'one', 'also', 'amazon', 'use', 'like']
    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in tokens if word not in stop_words and len(word) > 2]

# --- Main Page ---
def review_analysis_page():
    """Creates the content for the customer review analysis page."""
    st.title("🗣️ Customer Review Insights")
    st.markdown("Explore customer sentiment through ratings and review content. What do high ratings and popularity tell us? What are customers talking about?")

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

    # --- Rating vs. Popularity ---
    st.header("How Do Ratings Correlate with Popularity?")
    st.markdown("This chart investigates whether a higher rating corresponds to a higher number of reviews (`rating_count`).")

    col_graph_1, col_data_1 = st.columns([2,1]) # Reversed order
    with col_graph_1:
        fig_rating_pop = px.scatter(
            filtered_data,
            x="rating",
            y="rating_count",
            opacity=0.5,
            title="Ratings vs. Popularity (Number of Reviews)",
            labels={"rating": "Average Rating", "rating_count": "Number of Ratings"},
            color="rating",
            color_continuous_scale=px.colors.sequential.Viridis,
            hover_data=['product_name']
        )
        st.plotly_chart(fig_rating_pop, use_container_width=True)
    with col_data_1:
        st.markdown("##### Descriptive Statistics")
        st.dataframe(filtered_data[['rating', 'rating_count']].describe())
    
    rating_pop_corr = filtered_data['rating'].corr(filtered_data['rating_count'])
    st.markdown(f"""
    The scatter plot shows a weak positive correlation (coefficient: `{rating_pop_corr:.2f}`) between product ratings and the number of reviews. 
    While popular products tend to maintain good ratings (often above 4.0), a high rating alone doesn't guarantee widespread popularity.
    """)

    # --- Word Cloud from Reviews ---
    st.header("Most Frequent Words in Customer Reviews")
    st.markdown("This word cloud visualizes the most common terms found in the `review_content`, highlighting key themes and customer feedback.")

    # Handle potential missing reviews
    review_text = filtered_data['review_content'].dropna()
    
    if not review_text.empty:
        # Generate keywords
        keywords = review_text.apply(extract_keywords)
        all_keywords = [kw for sublist in keywords for kw in sublist]
        
        if all_keywords:
            text = ' '.join(all_keywords)
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='cividis', max_words=100).generate(text)
            
            col_graph_2, col_data_2 = st.columns([2,1]) # Reversed order
            with col_graph_2:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            with col_data_2:
                st.markdown("##### Top Review Keywords")
                keyword_counts = pd.Series(all_keywords).value_counts().head(10).reset_index()
                keyword_counts.columns = ['Keyword', 'Frequency']
                st.dataframe(keyword_counts)
            
            st.markdown("""
            The word cloud visually represents the most common words used in customer reviews. 
            Larger words indicate higher frequency, revealing key aspects of products that customers frequently discuss, both positive and negative.
            """)
        else:
            st.warning("Could not generate keywords from review content.")
    else:
        st.warning("No review content available to generate a word cloud.")


if __name__ == "__main__":
    review_analysis_page()
