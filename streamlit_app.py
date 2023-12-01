import os
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from padelpy import padeldescriptor

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

# Resampling and splitting data into train and test set
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Loading ML libraries
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# Page configuration
st.set_page_config(
  page_title='TrustTracker',
  page_icon='👌',
  initial_sidebar_state='expanded')

# Session state
if 'smiles_input' not in st.session_state:
  st.session_state.smiles_input = ''

# Utilities
if os.path.isfile('molecule.smi'):
  os.remove('molecule.smi') 
  
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


# The App    
st.title('TrustTracker 👌')
st.markdown('Welcome to TrustTracker! The application where you easily can check the quality, price, service and delivery of your favorite companies.')

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['About', 'Traditional Sentiment Analysis', 'Advanced Sentiment Analysis', 'Model performance', 'Dataset', 'Visualisations'])


with tab1:
  coverimage = Image.open('PARP1pred.jpg')
  st.image(coverimage)

        
with tab2:
  if st.session_state.smiles_input == '':
    
    with st.form('my_form'):
      st.subheader('Predict PARP1 inhibitory activity')

      smiles_txt = st.text_input('Enter SMILES notation', st.session_state.smiles_input)
      st.session_state.smiles_input = smiles_txt

      with st.expander('Example SMILES'):
        st.code('O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N1CCN(C(=O)C2CC2)CC1')
      
      submit_button = st.form_submit_button('Submit')

      
      
      if submit_button:
        st.subheader('⚛️ Input molecule:')
        with st.expander('Show SMILES', expanded=True):
          #st.write('**SMILES**')
          st.text(st.session_state.smiles_input)

        with st.expander('Show chemical structures', expanded=True):
          #st.write('**Chemical structure**')
          smi = Chem.MolFromSmiles(st.session_state.smiles_input)
          Chem.Draw.MolToFile(smi, 'molecule.png', width=900)
          mol_image = Image.open('molecule.png')
          st.image(mol_image)

      # Input SMILES saved to file
      f = open('molecule.smi', 'w')
      f.write(f'{st.session_state.smiles_input}\tmol_001')
      f.close()


      # Compute PADEL descriptors
      if st.session_state.smiles_input != '':
        st.subheader('🔢 Descriptors')
        if os.path.isfile('molecule.smi'):
          padeldescriptor(mol_dir='molecule.smi', 
                            d_file='descriptors.csv',
                            descriptortypes='data/PubchemFingerprinter.xml', 
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)

        descriptors = pd.read_csv('descriptors.csv')
        descriptors.drop('Name', axis=1, inplace=True)

        with st.expander('Show full set of descriptors as calculated for query molecule'):
          #st.write('**Full set of descriptors (calculated for query molecule)**')
          st.write(descriptors)
          st.write(descriptors.shape)


      # Load descriptor subset used in trained model
      if st.session_state.smiles_input != '':
        model = pickle.load(open('data/oversampling_PubChem_RandomForestClassifier.pkl', 'rb'))
        pubchem_subset = model.feature_names_in_

        query_desc_1 = descriptors.columns.difference(pubchem_subset)
        query_desc_2 = descriptors.drop(query_desc_1, axis=1)

        with st.expander('Show subset of descriptors as used in trained model'):
          #st.write('**Subset of descriptors (used in trained model)**')
          st.write(query_desc_2)
          st.write(query_desc_2.shape)


      # Read in saved classification model
      if st.session_state.smiles_input != '':
        st.subheader('🤖 Predictions')
        pred = int(model.predict(query_desc_2))
        if pred == 0:
          st.error('Inactive')
        if pred == 1:
          st.success('Active')
with tab3:
  st.header('What is PARP1?')
  st.write('Poly (ADP-ribose) polymerase-1 (PARP-1) is an enzyme that catalyzes the ADP-ribosylation of a specific protein and plays a vital role in DNA repair. It has become an attractive target as inhibition of PARP-1 causes a toxic accumulation of DNA double strand breaks in cancer cells, particularly those with BRCA1/2 deficiency, which are found in breast, ovarian, prostate, and pancreatic cancers.')
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
