from cmath import log
from pydoc import html
import streamlit as st
from streamlit import components
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
import sklearn
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
plt.style.use('seaborn')

st.title('Fake News Model')

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('./data/clean_data.zip')
    with open('model/logistic_model.pickle', 'rb') as file:
        log_model = pickle.load(file)
    with open('model/label_encoder.pickle', 'rb') as file:
        label = pickle.load(file)
    return data, log_model, label

@st.cache
def get_prediction(article):
    prediction = log_model.predict(article)
    probability = log_model.predict_proba(article)
    prediction = label.inverse_transform(prediction)[0]
    return prediction, probability[0]

data_load_state = st.text('Loading data...')
data, log_model, label = load_data()
data_load_state.text("Done! (using st.cache)")

def get_query(query):
    data, log_model, label = load_data()
    query_data = data.copy()
    query_list = query.split()
    for idx, word in enumerate(query_list):
        if idx == 0:
            results = query_data[query_data['cleaned_title'].str.contains(word)]
        else:
            results = results[results['cleaned_title'].str.contains(word)]   
        if results.empty:
            print('No matching results')
            return results
    return results

st.subheader('What Is This?')
st.markdown(
    'This is a live version of a fake news classification model that attempts to identify fake news by article content \n.' 
    'For more information on its construction go to https://github.com/nonlocal-lia/fake_news_detector. \n'
    'Below you can select an article by index or title and see the models predictions \n'
    'as well as examine what features of the text the model was using in arriving at its predictions with the help of lime. \n'
    'Fake news as defined in the data does not mean all the claims within are false, nor does real news mean the claims are true. \n'
    'Fake news in the data tended to be inflammatory or clickbaiting stories with poor sourcing or pure opinion, and real news was typically news from mainstream outlets. \n'
    'This is not a truth and falsehood detector.'
)

st.subheader('Select Article')
option = st.selectbox(
     'How would you like to select articles?',
     ('index in database', 'search titles in database', 'url'))
if option=='index in database':
    article_idx = st.slider('Article Index:', 0, data.shape[0])
    article = [data['cleaned_text'][article_idx]]
    prediction, probability = get_prediction(article)
if option=='search titles in database':
    query = st.text_input('Search Titles', value='input title')
    if query != 'input title' and query != '':
        results = get_query(query)
        if not results.empty:
            st.write('Found {} results'.format(results.shape[0]))
            if results.shape[0] > 5:
                first_five = tuple(results['cleaned_title'][:5])
                title = st.selectbox(
                    'Select from first five matches:',
                    first_five)
                article = data[data['cleaned_title']==title]['cleaned_text']
                article_idx = data[data['cleaned_title']==title].index[0]
                prediction, probability = get_prediction(article)
            else:
                first_results = tuple(results['cleaned_title'])
                title = st.selectbox(
                    'Select from matches:',
                    first_results)
                article = data[data['cleaned_title']==title]['cleaned_text']
                article_idx = data[data['cleaned_title']==title].index[0]
                prediction, probability = get_prediction(article)
        else:
            prediction, probability = None, None
    else:
        st.write("no input, try again")
        prediction, probability = None, None
if option=='url':
    st.write('Not implimented yet')
    prediction, probability = None, None

if prediction:
    st.subheader('Prediction:')
    if prediction == 'fake':
        st.write('Article is a {pred} news story with {prob:.1f}% probability'.format(pred=prediction, prob=probability[0]*100))
    if prediction == 'real':
        st.write('Article is a {pred} news story with {prob:.1f}% probability'.format(pred=prediction, prob=probability[1]*100))
    
    explainer = LimeTextExplainer(class_names=label.classes_)
    exp_log = explainer.explain_instance(data['cleaned_text'][article_idx], log_model.predict_proba, labels=[1])
    html_lime = exp_log.as_html(text=data['cleaned_text'][article_idx])

    st.subheader('Lime Explanation')
    components.v1.html(html_lime, width=1100, height=350, scrolling=True)

st.subheader('LDA Topic Clustering of Data')
st.markdown(
    'The analyzed articles can be clustered into 5 categories using LDA which you can explore below:\n'
    '* Category 1: Movie/TV Celebrities \n'    
    '* Category 2: US Government \n'
    '* Category 3: US Political Campaigns \n'
    '* Category 4: Foreign and Misc News \n'
    '* Category 5: Music Celebrities \n'
)

with open('./lda.html', 'r') as f:
    html_string = f.read()
components.v1.html(html_string, width=1300, height=800, scrolling=False)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Model constructed by <a href="https://github.com/nonlocal-lia/fake_news_detector" >Lia Elwonger </a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)