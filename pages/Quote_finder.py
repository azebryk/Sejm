# Import libraries
import streamlit as st
from streamlit_tags import st_tags
import pandas as pd

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True)
def load_data(file_name):
    df = pd.read_parquet(file_name)
    df.columns = ['meeting', 'speaker', 'speech', 'date', 'party']
    # df = df[df['speech_len'] != 0]
    return df

def query_df(df, phrase):
    # phrase = phrase.lower()
    return df.query(f'speech.str.contains("{phrase}")', engine='python')

# Page Titles
col1, col2 = st.columns([4,2])
col1.title('Polish Parliament Speeches')
col2.image('images/sejm_logo.jpg', width=200)

st.title('Quote Finder')

# Loading data
df = load_data('./data/speeches.parquet')


keyword = st_tags(
    label='## Please enter the text you want to search within speeches',
    text='Press enter to add more',
    value=['TVN'],
    suggestions=['pandemia', 'covid'],
    maxtags=6,
    key='1')

df_filter = df.copy()
if len(keyword) > 0:
    for x in range(len(keyword)):
        df_result = query_df(df_filter, keyword[x])
        df_filter = df_result

    st.write(f'#### Found {df_result.shape[0]} results.')
    st.dataframe(df_result)
else:
    st.write(f'#### Found {df.shape[0]} results.')
    st.dataframe(df)


st.write('If you want to see the full context of speech you can check it on based on '
         'https://www.sejm.gov.pl/Sejm9.nsf/stenogramy.xsp')

