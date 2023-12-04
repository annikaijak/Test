import os
import pickle
from PIL import Image

# For building application
import streamlit as st

# Basic data manipulation:
import numpy as np
import pandas as pd

# Visualisations:
import seaborn as sns
import matplotlib.pyplot as plt

# NLP
import spacy
nlp = spacy.load('en_core_web_sm')

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# Page configuration
st.set_page_config(
  page_title='TrustTracker',
  page_icon='👌',
  initial_sidebar_state='expanded')
  
# Function to load the dataset
@st.experimental_memo
def load_data():
    # Define the file path
    file_path = 'https://raw.githubusercontent.com/MikkelONielsen/TrustTracker/main/trust_pilot_reviews_data_2022_06.csv'
    
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)
    
    return df


# Load the data using the defined function
df = load_data()

pipe_svm = pickle.load(open('data/model.pkl', 'rb'))

# Defining functions
def text_prepro(texts: pd.Series) -> list:
  # Creating a container for the cleaned texts
  clean_container = []
  # Using spacy's nlp.pipe to preprocess the text
  for doc in nlp.pipe(texts):
    # Extracting lemmatized tokens that are not punctuations, stopwords or
    # non-alphabetic characters
    words = [words.lemma_.lower() for words in doc
            if words.is_alpha and not words.is_stop and not words.is_punct]

    # Adding the cleaned tokens to the container "clean_container"
    clean_container.append(" ".join(words))

  return clean_container

# Defining the ML function
def predict(placetext):
  text_ready = []
  text_ready = text_prepro(pd.Series(placetext))
  result = pipe_svm.predict(text_ready)
  if result == 0:
    return "Negative sentiment"
  if result == 1:
    return "Positive sentiment"

categories = {
    "Price": [
        "price", "cost", "expensive", "cheap", "value", "pay", "affordable",
        "pricey", "budget", "charge", "fee", "pricing", "rate", "worth", "economical"
    ],
    "Delivery": [
        "deliver", "delivery", "shipping", "dispatch", "courier", "ship", "transit",
        "postage", "mail", "shipment", "logistics", "transport", "send", "carrier", "parcel"
    ],
    "Quality": [
        "quality", "material", "build", "standard", "durability", "craftsmanship",
        "workmanship", "texture", "construction", "condition", "grade", "caliber",
        "integrity", "excellence", "reliability", "sturdiness", "performance"
    ]
}

def lemmatize_keywords(categories):
    lemmatized_categories = {}
    for category, keywords in categories.items():
        lemmatized_keywords = [nlp(keyword)[0].lemma_ for keyword in keywords]
        lemmatized_categories[category] = lemmatized_keywords
    return lemmatized_categories
    
list_lab = []
def categorize_review(text_review):
    lemmatized_review = " ".join([token.lemma_ for token in nlp(text_review.lower())])
    for category, keywords in lemmatize_keywords(categories).items():
        if any(keyword in lemmatized_review for keyword in keywords):
          list_lab.append(category)
    return list_lab
    if len(list_lab) == 0:
      return "Other"

def classifier(text):
  category = categorize_review(text)
  sentiment = predict(text)
  return category, sentiment

# The App    
st.title('TrustTracker 👌')
st.markdown('Welcome to TrustTracker! The application where you easily can check the quality, price, service and delivery of your favorite companies.')

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['About', 'Traditional Sentiment Analysis', 'Transformer-Based Sentiment Analysis', 'Model performance', 'Dataset', 'Visualisations'])


with tab1:
  st.header("About the application")

        
with tab2:
    
  with st.form('my_form'):
    st.subheader('Traditional Approach')

    review_txt = st.text_input('Enter your review here')

    with st.expander('Example Review'):
      st.write("The price of the product was way too high. Would not recommend this company.")
      
    submit_button = st.form_submit_button('Submit')
      
    if submit_button:
      st.subheader('Sentiment Analysis of the review')
      result = classifier(review_txt)
      st.text(result)


with tab3:
  st.header('Transformer Approach')
  st.write('This tab includes a Transformer-Based Sentiment Analysis using RoBERTa.')

with tab4:
  st.header('Model performance')
  st.write('''
    In our work, we retrieved a human PARP-1 biological dataset from the ChEMBL database. The data was curated and resulted in a non-redundant set of 2,018 PARP-1 inhibitors, which can be divided into:
    - 1,720 active compounds
    - 298 inactive compounds
    ''')

with tab5:
  st.header('Dataset')
  # Display dataset overview
  st.subheader("Dataset Overview")
  st.dataframe(df.head())

with tab6:
  st.header('Visualisations')

  # Reviews by number of companies
  st.subheader("Reviews by Number of Companies")
  # Counting how many reviews each company has
  reviews_count = df['name'].value_counts()

  # Setting up the plot
  plt.figure(figsize=(10, 6))
  ax = reviews_count.value_counts().sort_index() \
  .plot(kind='bar',
        title='Count of Reviews by Number of Companies ',
        figsize=(10, 5))
  ax.set_xlabel('Number of Reviews')
  ax.set_ylabel('Number of Companies')
  st.pyplot(plt)  

  # Reviews by rating
  st.subheader("Reviews by Rating")
    
  plt.figure(figsize=(10, 6))
  ax = df['rating'].value_counts().sort_index() \
  .plot(kind='bar',
        title='Count of Reviews by Stars',
        figsize=(10, 5))
  ax.set_xlabel('Review Stars')
  st.pyplot(plt)

  # Reviews by year
  st.subheader("Reviews by Year")
    
  # Converting the reviewed_at column to datetime
  df['reviewed_at'] = pd.to_datetime(df['reviewed_at'])
    
  # We convert the reviewed_at column to a string format with only the year
  # count how many reviews were made each year, sort the index and reset the index
  reviews_per_day = df['reviewed_at'].dt.strftime('%Y').value_counts().sort_index().reset_index(name='counts')
    
  # Then we plot the figure
  plt.figure(figsize=(20,5))
  plt.bar(reviews_per_day['reviewed_at'], reviews_per_day['counts'])
  plt.title('Review count by year')
  plt.ylabel('Number of reviews')
  plt.xlabel('Year')
  st.pyplot(plt)

  # Reviews by user
  st.subheader("Reviews by User")
    
  # Counting how many reviews the authors have made, but only for the authers that
  # have made more than 3 reviews.
  reviews_per_author = df['author_name'].value_counts().loc[lambda x : x > 3].reset_index(name='counts')
    
  plt.figure(figsize=(15,6))
  plt.bar(reviews_per_author['author_name'], reviews_per_author['counts'])
  plt.title('Review count by author')
  plt.xticks(rotation=70)
  plt.yticks([])
  plt.ylabel('Number of Reviews')
  plt.xlabel('Review Author')
  st.pyplot(plt)
