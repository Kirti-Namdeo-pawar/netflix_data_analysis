import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Netflix data
netflix_data = pd.read_csv('netflix_titles.csv')


# title of dashboard
st.markdown("""
    <style>
    .main-heading {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #E50914;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .sub-heading {
        font-size: 18px;
        font-style: italic;
        text-align: center;
        color: #555;
    }
    </style>
    <h1 class="main-heading">✨ Netflix Content Insights Dashboard 🎬</h1>
    <p class="sub-heading">"Discover Trends, Insights, and Hidden Patterns in Netflix Data!"</p>
""", unsafe_allow_html=True)

# Toggle for Dark/Light Mode
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

# Apply CSS dynamically
if dark_mode:
    st.markdown("""
        <style>
            html, body, .stApp {
                background-color: #121212 !important;
                color: white !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, div {
                color: white !important;
            }
            /* Fix dropdown, multi-select, and slider */
            .stSelectbox, .stMultiSelect, .stSlider {
                background-color: #333 !important;
                color: white !important;
            }
            /* Fix dropdown items */
            .stMultiSelect div[role="listbox"], .stSelectbox div[role="listbox"] {
                background-color: #333 !important;
                color: white !important;
            }
            /* Fix slider */
            .stSlider .st-br {
                background-color: red !important;
            }
            /* Fix sidebar */
            .stSidebar {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            /* Fix input fields */
            .stTextInput, .stNumberInput {
                background-color: #222 !important;
                color: white !important;
                border: 1px solid white !important;
            }
            /* Fix button */
            .stButton > button {
                background-color: red !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            html, body, .stApp {
                background-color: white !important;
                color: black !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, div {
                color: black !important;
            }
            .stSelectbox, .stMultiSelect, .stSlider {
                background-color: white !important;
                color: black !important;
            }
            .stMultiSelect div[role="listbox"], .stSelectbox div[role="listbox"] {
                background-color: white !important;
                color: black !important;
            }
            .stSidebar {
                background-color: #f0f2f6 !important;
                color: black !important;
            }
            .stTextInput, .stNumberInput {
                background-color: white !important;
                color: black !important;
                border: 1px solid black !important;
            }
            .stButton > button {
                background-color: blue !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header('Filter Options')

# Type Filter
type_options = netflix_data['type'].unique().tolist()
selected_type = st.sidebar.multiselect('Select Type', type_options, default=type_options)

# Year Filter
min_year = st.sidebar.slider('Minimum Release Year', min(netflix_data['release_year']), max(netflix_data['release_year']), min(netflix_data['release_year']))
max_year = st.sidebar.slider('Maximum Release Year', min(netflix_data['release_year']), max(netflix_data['release_year']), max(netflix_data['release_year']))

# Rating Filter
rating_options = netflix_data['rating'].dropna().unique().tolist()
selected_rating = st.sidebar.multiselect('Select Rating', rating_options, default=rating_options)

# Country Filter
country_options = netflix_data['country'].dropna().unique().tolist()
selected_country = st.sidebar.multiselect('Select Country', country_options, default=country_options[:10])  # Limit default selection

# Genre Filter
category_options = netflix_data['listed_in'].dropna().unique().tolist()
selected_category = st.sidebar.multiselect('Select Category', category_options, default=category_options[:5])

# Apply Filters
filtered_data = netflix_data[
    (netflix_data['type'].isin(selected_type)) &
    (netflix_data['release_year'] >= min_year) &
    (netflix_data['release_year'] <= max_year) &
    (netflix_data['rating'].isin(selected_rating)) &
    (netflix_data['country'].isin(selected_country)) &
    (netflix_data['listed_in'].isin(selected_category))
]

# Tabs for Dashboard
tab1, tab2, tab3, tab4, tab5 ,tab6 = st.tabs(["Overview", "Trends", "Directors", "Actors", "Correlation","Sentiment Analysis"])

with tab1:
    st.subheader("Filtered Data")
    st.write(filtered_data)

with tab2:
    st.subheader("Trends in Netflix Shows")
    
    # Countplot of release years
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_data, x='release_year', palette='viridis', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)
    
    # Line plot of shows per year
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=filtered_data['release_year'].value_counts().sort_index(), marker='o', color='orange', ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.subheader("Top 10 Directors")
    top_directors = filtered_data['director'].value_counts().head(10)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_directors.values, y=top_directors.index, palette='magma', ax=ax3)
    st.pyplot(fig3)

with tab4:
    st.subheader("Top 10 Actors")
    filtered_data['cast_count'] = filtered_data['cast'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
    top_actors = filtered_data.explode('cast')['cast'].value_counts().head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_actors.values, y=top_actors.index, palette='coolwarm', ax=ax4)
    st.pyplot(fig4)

with tab5:
    st.subheader("Correlation Heatmap")
    filtered_data['listed_in_count'] = filtered_data['listed_in'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
    numeric_data = filtered_data[['release_year', 'cast_count', 'listed_in_count']]
    
    if not numeric_data.empty and len(numeric_data.columns) > 1:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax5)
        st.pyplot(fig5)
    else:
        st.write("Not enough numeric data available for correlation heatmap.")

with tab6:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import plotly.express as px

    # Download VADER for sentiment analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    # Function to get sentiment category
    def get_sentiment(text):
        if pd.isna(text):
            return "Neutral"
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Apply sentiment analysis
    netflix_data['sentiment'] = netflix_data['description'].apply(get_sentiment)

    # Sentiment Distribution Plot
    st.subheader("Sentiment Analysis on Show Descriptions")
    sentiment_counts = netflix_data['sentiment'].value_counts()

    fig = px.pie(values=sentiment_counts.values, 
                names=sentiment_counts.index, 
                title="Sentiment Distribution of Netflix Show Descriptions",
                color=sentiment_counts.index,
                color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig)
