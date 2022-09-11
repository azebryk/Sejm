
# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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
    df = df[df['speech_len'] != 0]
    return df

def query_df(phrase):
    # phrase = phrase.lower()
    return df.query(f'speech_clean_end_4.str.contains("{phrase}")', engine='python')

def generate_personal_cloud(speaker):
    # words = df[df['speaker'] == speaker]['lemma_words'].sum()
    words = df[df['speaker'] == speaker]['lemma_words'].apply(lambda x: list(x)).sum()
    words = ' '.join(words)
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = add_stop_words,     # mask = pic,
                min_font_size = 10).generate(words)

    # Create a figure of the generated cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    # col1.pyplot()




# Loading data
df = load_data('./data/speeches_lemma.parquet')


# Page Titles
col1, col2 = st.columns([4,2])
col1.title('Polish Parliament Speeches')
col2.image('images/sejm_logo.jpg', width=200)

# Project overview and business usage
col_1, col_2 = st.columns(2)
col_1.subheader("Project Overview")
col_1.write('This app allows you to investigate polish parliament speeches in depth.\n'
            '\n I scraped, cleaned and preprocessed'
         ' data from https://www.sejm.gov.pl/ to make it more useful for analysis.\n'
         '\n **Github repo:** https://github.com/azebryk/Sejm\n'
         '\n **Contact:** https://www.linkedin.com/in/agata-zebryk/')

col_2.subheader("Business usage")
col_2.write('This app might be useful for political journalist or anyone who is interested in politics.\n'
            '\n Do you remember interesting quote from polish parliament, but you are not sure who said this and when?\n'
            '\n Have you ever wonder what are the main topics in polish parliament?\n'
         '\n If so, you are in the right place :)')

st.subheader('Application Content:')
st.write('- Exploratory Data Analysis\n'
         '\n- Personalized Word Cloud'
         '\n- Quote Finder'
         '\n- Topic Modeling')

# General info
st.subheader('General Information')
col1, col2, col3, col4, col5 = st.columns(5)
col1.write(f'**Number of meetings:** {df["date"].nunique()}')
col2.write(f'**Start Date:** {df["date"].min()}')
col3.write(f'**End Date:** {df["date"].max()}')
col4.write(f'**Number of unique speakers**: {df["speaker"].nunique()}')
col5.write(f'**Number of speeches**: {df.shape[0]}')
# Exploratory Data Analysis
st.subheader('Exploratory Data Analysis')

## Speaker group
st.write('#### Speaker group')
col1, col2  = st.columns(2)
speaker_group = df['speaker_group'].value_counts()
fig = px.pie(values= speaker_group.values, names= speaker_group.index, hole=.3)
fig.update_layout({'title':{'text': '<b>Speeches per speaker group<b>', 'x':0.5}})
col1.plotly_chart(fig)

col1.write('**There are 3 main groups of speakers in polish parliament:**\n'
         '\n -*Marszałek/Wiecemarszałek* - moderator\n'
         '\n -*Poseł* - member\n'
         '\n -*Other*')

# col2.write('#### Top speakers')
fig = px.box(df, x='speech_len', color='speaker_group', width=800)
fig.update_xaxes(range=[0, 15000])
fig.update_layout({'title':{'text': '<b>Length of speeches per speaker group<b>', 'x':0.5}},
                   yaxis_title="Speaker Group",
                  xaxis_title="Speech length")
col2.plotly_chart(fig)
col2.write('**Comment:**')
col2.write('- As expected Marszałek/Wice, who moderates parliament meetings, have usually very short speeches like "Thank you, Next"\n'
'\n - Than we have "Poseł" and definitely the longest speeches comes from "Other" group, who are "guests" in parliaments meeting like president or ministers')

st.write('#### Speeches count')
fig = px.histogram(df.groupby('date')['speech'].count().values, nbins=20)
fig.update_layout({'title':{'text': '<b>Distribution of speeches count per meeting<b>', 'x':0.5}},
                  showlegend=False,
                  xaxis_title="Speeches per meeting")
st.plotly_chart(fig)

st.write('#### Speeches count per party')
speeches_party = df[df['speaker_group'] == 'Poseł']['party'].value_counts()
fig = px.bar(y= speeches_party.values, x= speeches_party.index, color = speeches_party.index, width=1000, text_auto='.3s')
fig.update_layout({'title':{'text': '<b>Speeches count per Political Party<b>', 'x':0.5},
                   'xaxis_title': {'text': 'Political Party'},
                   'yaxis_title': {'text': 'Speeches count'}},
                  showlegend=False)
st.plotly_chart(fig)
# st.write(df)
#f
#
