import streamlit as st
import pandas as pd
import json
from transformers import BartForConditionalGeneration, BartTokenizer

# Set the page layout to wide
st.set_page_config(layout="wide")
##############################################################################################


# Cache the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to summarize an article
def summarize_article(text):
    inputs = tokenizer(text, max_length=1000, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



##############################################################################################
# Load the data
with open('trial.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the categories
categories = ["India", "World", "Business", "Entertainment", "Technology", "Sports", "Science"]

# Map categories to Source Names for India
india_sources = ["google-news-in", "the-times-of-india", 'the-hindu']

st.title("News Dashboard")

# Create tabs for each category
tabs = st.tabs(categories)

# Function to display news articles in a grid
def display_news(articles):
    num_columns = 2
    for i in range(0, len(articles), num_columns):
        cols = st.columns(num_columns)
        for col, article in zip(cols, articles[i:i + num_columns]):
            if isinstance(article, dict):  # Ensure article is a dictionary
                col.markdown(f"**{article.get('title', 'No Title')}**")
                col.markdown(f"*{article.get('author', 'Unknown Author')}*")
                if col.button("Summarize", key=f"summarize_{article['url']}"):
                    summary = summarize_article(article.get('content', ''))
                    col.markdown(summary)
                col.markdown(f"[Read more]({article.get('url', '#')})")
                col.markdown("---")

# Display news articles in each tab
for category, tab in zip(categories, tabs):
    with tab:
        st.header(f"{category} News")

        if category == "India":
            # Filter the DataFrame based on source names for India
            filtered_df = df[df['id'].isin(india_sources)]
        else:
            # Filter the DataFrame based on category for other segments
            filtered_df = df[df['description'].str.contains(category, case=False, na=False)]

        st.write(f"Filtered {len(filtered_df)} rows for category {category}")

        # Collect all articles from the filtered DataFrame
        articles = []
        for _, row in filtered_df.iterrows():
            st.write(f"Processing row with id {row['id']}")
            if isinstance(row['data'], str):
                try:
                    articles.extend(json.loads(row['data'].replace("'", '"')))
                except json.JSONDecodeError as e:
                    st.write(f"JSONDecodeError: {e}")
                    continue
            elif isinstance(row['data'], list):
                articles.extend(row['data'])

        st.write(f"Collected {len(articles)} articles for category {category}")

        if articles:
            display_news(articles)
        else:
            st.write("No news articles available for this category.")
