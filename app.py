import streamlit as st
import pandas as pd
import json
from transformers import BartTokenizer, BartForConditionalGeneration, MBartForConditionalGeneration, MBart50Tokenizer

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the data
with open('trial.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the categories
categories = ["India", "World", "Business", "Entertainment", "Technology", "Sports", "Science"]

# Map categories to Source Names for India
india_sources = ["google-news-in", "the-times-of-india", "the-hindu"]

# Cache the models and tokenizers
@st.cache_resource
def load_models():
    bart_model_name = "facebook/bart-large-cnn"
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
    
    mbart_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    mbart_tokenizer = MBart50Tokenizer.from_pretrained(mbart_model_name)
    mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
    mbart_tokenizer.src_lang = "en_XX"
    return bart_model, bart_tokenizer, mbart_model, mbart_tokenizer

bart_model, bart_tokenizer, mbart_model, mbart_tokenizer = load_models()

st.title("News Dashboard")

# Initialize session state for summaries and translations
if 'summaries' not in st.session_state:
    st.session_state['summaries'] = {}

if 'translations' not in st.session_state:
    st.session_state['translations'] = {}

# Function to summarize an article
def summarize_article(text):
    inputs = bart_tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to translate text to Hindi
def translate_to_hindi(text):
    inputs = mbart_tokenizer(text, return_tensors='pt', truncation=True)
    translated_ids = mbart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True, forced_bos_token_id=mbart_tokenizer.lang_code_to_id["hi_IN"])
    translation = mbart_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translation

# Function to display news articles in a grid
def display_news(articles, category):
    num_columns = 2
    for i in range(0, len(articles), num_columns):
        cols = st.columns(num_columns)
        for j, (col, article) in enumerate(zip(cols, articles[i:i + num_columns])):
            if isinstance(article, dict):  # Ensure article is a dictionary
                article_id = article['url']
                col.markdown(f"**{article.get('title', 'No Title')}**")
                col.markdown(f"*{article.get('author', 'Unknown Author')}*")
                
                # Unique keys for each button by including category, article_id, and index
                summarize_key = f"{category}_summarize_{i}_{j}_{article_id}"
                translate_key = f"{category}_translate_{i}_{j}_{article_id}"
                
                if summarize_key not in st.session_state['summaries']:
                    if col.button("Summarize", key=summarize_key):
                        summary = summarize_article(article.get('content', ''))
                        st.session_state['summaries'][summarize_key] = summary
                        st.session_state['translations'][summarize_key] = None
                        
                if summarize_key in st.session_state['summaries']:
                    col.markdown(st.session_state['summaries'][summarize_key])
                    if translate_key not in st.session_state['translations'] or st.session_state['translations'][translate_key] is None:
                        if col.button("Translate", key=translate_key):
                            translation = translate_to_hindi(st.session_state['summaries'][summarize_key])
                            st.session_state['translations'][translate_key] = translation
                            
                if translate_key in st.session_state['translations'] and st.session_state['translations'][translate_key] is not None:
                    col.markdown(st.session_state['translations'][translate_key])

                col.markdown(f"[Read more]({article.get('url', '#')})")
                col.markdown("---")

# Display news articles in each tab
for category, tab in zip(categories, st.tabs(categories)):
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
            display_news(articles, category)  # Pass the category to ensure unique keys
        else:
            st.write("No news articles available for this category.")
