
# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
# from st_aggrid import AgGrid
# from st_aggrid.grid_options_builder import GridOptionsBuilder
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_data(file_name):
    df = pd.read_parquet(file_name)
    return df

def query_df(phrase):
    # phrase = phrase.lower()
    return df.query(f'speech_clean_end_4.str.contains("{phrase}")', engine='python')

def generate_personal_cloud(speaker):
    # words = df[df['speaker'] == speaker]['lemma_words'].sum()
    pic = np.array(Image.open('images/polska.jpg'))
    words = df[df['speaker'] == speaker]['lemma_words'].apply(lambda x: list(x)).sum()
    words = ' '.join(words)
    wordcloud = WordCloud(#width = 500, height = 500,
                background_color ='white',
                stopwords = add_stop_words,  mask = pic,# min_font_size = 10
        ).generate(words)


    # Create a figure of the generated cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    col1.pyplot()


df = load_data('./data/speeches_lemma.parquet')
add_stop_words = ['marszałek', 'izbo', 'izba', 'zatem', 'wysoki', 'mieć', 'móc', 'być', 'chcieć', 'dziękować',
                  'poseł', 'by', 'mówić', 'powiedzieć', 'szanowni', 'państwo']

# Page Titles
col1, col2 = st.columns([4,2])
col1.title('Polish Parliament Speeches')
col2.image('images/sejm_logo.jpg', width=200)


# Personalized Word Cloud
st.title('Personalized Word Cloud')
st.write('- A word cloud is a word visualization that displays words proportional to how often they appear in a text.\n '
         '\n- To provide meaningful insights I dropped polish stop words (like *że*, *żeby*) and other useless phrases (*Wysoka Izbo*).\n'
         '\n- Each word has been lemmatized -  converted into its base form.' )


speaker = st.selectbox('Please select speaker', df['speaker'].unique(), index=list(df['speaker'].unique()).index('Poseł Grzegorz Braun'))
col1, col2 = st.columns([4,2])
generate_personal_cloud(speaker)
col2.subheader(f"Number of speeches: {df[df['speaker'] == speaker].shape[0]}")

