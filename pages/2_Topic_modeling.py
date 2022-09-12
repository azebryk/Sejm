
# Import libraries
import streamlit as st
from streamlit import components
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# Settings
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_data(file_name):
    df = pd.read_parquet(file_name)
    return df

@st.cache(allow_output_mutation=True)
def generate_lda_model(num_topics):
    data_bigrams_trigrams = df['data_bigram_trigrams'].apply(lambda x: list(x))
    corpus = df_corpus['corpus'].apply(lambda x: list(x))
    id2word = corpora.Dictionary(data_bigrams_trigrams)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=42,
                                                update_every=1,
                                                chunksize=50,  # 50
                                                passes=10,
                                                alpha="auto")
    return id2word, corpus, lda_model

df = load_data('./data/bigrams_gensim.parquet')
df_corpus = load_data('./data/corpus.parquet')
# Page Titles
col1, col2 = st.columns([4,2])
col1.title('Polish Parliament Speeches')
col2.image('images/sejm_logo.jpg', width=200)

st.title('Topic Modeling')
st.write('Topic modeling is an unsupervised machine learning technique that’s capable of scanning a set of documents, '
         'detecting word and phrase patterns within them, and automatically clustering word groups and similar '
         'expressions that best characterize a set of documents.\n'
         '\nI used one of the most popular techniques of topic modeling -  **Latent Dirichlet Allocation**. Key concepts:\n'
         '\n- LDA extracts topics from a corpus of documents. The number of topics has to be previously defined by us.\n'
         '\n- For each document, LDA assignes topic/topics as percentages defining how much this document is about those topics\n'
         '**Here is a visualation of 10 topics built based on all parliament speeches:**')

# st.write(df)


# st.write(data_bigrams_trigrams[0][:20])

#TF-IDF REMOVAL

# texts = data_bigrams_trigrams
# corpus = [id2word.doc2bow(text) for text in texts]
# tfidf = TfidfModel(corpus, id2word=id2word)
# low_value = 0.03
# words  = []
# words_missing_in_tfidf = []
# for i in range(0, len(corpus)):
#     bow = corpus[i]
#     low_value_words = [] #reinitialize to be safe. You can skip this.
#     tfidf_ids = [id for id, value in tfidf[bow]]
#     bow_ids = [id for id, value in bow]
#     low_value_words = [id for id, value in tfidf[bow] if value < low_value]
#     drops = low_value_words+words_missing_in_tfidf
#     for item in drops:
#         words.append(id2word[item])
#     words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing
#
#     new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
#     corpus[i] = new_bow

# print(type(corpus))
# print(len(corpus))
# df['corpus'] = corpus
# df[['date', 'corpus']].to_parquet('corpus.parquet')
# LDA Model

id2word, corpus, lda_model = generate_lda_model(10)
# st.write(corpus[0][:5])

vis = gensimvis.prepare(lda_model, corpus, id2word)
html_string = pyLDAvis.prepared_data_to_html(vis)
components.v1.html(html_string, width=1300, height=800)