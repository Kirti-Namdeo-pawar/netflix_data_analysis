import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Load the Netflix data
netflix_data = pd.read_csv('netflix_titles.csv')  # Replace 'netflix_data.csv' with the actual path to your dataset

# Sidebar for filtering options
st.sidebar.header('Filter Options')

# Filter by type (Movie or TV Show)
type_options = netflix_data['type'].unique().tolist()
selected_type = st.sidebar.multiselect('Select Type', type_options, default=type_options)

# Filter by release year range
min_year = st.sidebar.slider('Minimum Release Year', min(netflix_data['release_year']), max(netflix_data['release_year']), min(netflix_data['release_year']))
max_year = st.sidebar.slider('Maximum Release Year', min(netflix_data['release_year']), max(netflix_data['release_year']), max(netflix_data['release_year']))
rating_options = netflix_data['rating'].dropna().unique().tolist()
selected_rating = st.sidebar.multiselect('Select Rating', rating_options, default=rating_options)
# Filter by country
#country_options = netflix_data['country'].dropna().unique().tolist()
#selected_country = st.sidebar.multiselect('Select Country', country_options, default=country_options)
#category=netflix_data['listed_in'].dropna().unique().tolist()
#selected_category=st.sidebar.multiselect('Select Category',category,default=category)
# Filtered data based on selected options
filtered_data = netflix_data[
    (netflix_data['type'].isin(selected_type)) &
    (netflix_data['release_year'] >= min_year) &
    (netflix_data['release_year'] <= max_year) &
     (netflix_data['rating'].isin(selected_rating)) 
  
  
]

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_data)

# Visualization 1: Countplot of release years
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(data=filtered_data, x='release_year', palette='viridis', ax=ax1)
ax1.set_xlabel('Release Year')
ax1.set_ylabel('Number of Shows')
ax1.set_title('Number of Shows Released Each Year')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
st.pyplot(fig1)

# Visualization 2: Line plot of number of shows over release years
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=filtered_data['release_year'].value_counts().sort_index(), marker='o', color='orange', ax=ax2)
ax2.set_xlabel('Release Year')
ax2.set_ylabel('Number of Shows')
ax2.set_title('Number of Shows Over Release Years')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
st.pyplot(fig2)

# Visualization 3: Histogram of release years
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.histplot(data=filtered_data, x='release_year', bins=len(filtered_data['release_year'].unique()), kde=True, color='skyblue', ax=ax3)
ax3.set_xlabel('Release Year')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Release Years')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
st.pyplot(fig3)

# Additional Visualization: Bar plot of top 10 directors
top_directors = filtered_data['director'].value_counts().head(10)
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='magma', ax=ax4)
ax4.set_xlabel('Number of Shows Directed')
ax4.set_ylabel('Director')
ax4.set_title('Top 10 Directors')
st.pyplot(fig4)

show_category = netflix_data['listed_in'].value_counts().head(10)
fig5, ax5 = plt.subplots(figsize=(10, 6))
show_category.plot(kind='bar',title='Top categories of shows',color=['lightgreen','orange','violet'])
plt.xlabel('Categories')
plt.ylabel('Number of shows')

plt.savefig("topShowCategories.png")
st.pyplot(fig5)


# Adding numeric features for correlation
filtered_data['cast_count'] = filtered_data['cast'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
filtered_data['listed_in_count'] = filtered_data['listed_in'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

# Select only numeric columns for the heatmap
numeric_data = filtered_data[['release_year', 'cast_count', 'listed_in_count']]

# Check if there are enough numeric columns for correlation
if not numeric_data.empty and len(numeric_data.columns) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation = numeric_data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
else:
    st.write("Not enough numeric data available for correlation heatmap.")
