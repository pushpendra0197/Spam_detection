import numpy as np
import pandas as pd
nltk.download("stopwords")
from nltk.corpus import stopwords
import spacy
import re
from nltk.stem import PorterStemmer
import string
from string import punctuation
pos_stem=PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import streamlit as st
cv=CountVectorizer()
data=joblib.load("clean_data_forfit")
cv.fit(data)
st.title("Spam Detection{messages and mails}")
input_text=st.text_input("Enter your Mail subject or message")
button=st.button("Spam Check")
if button:
    def clean(data):
       review=data.lower()
       review=review.split()    
       review=[ i for i in review if i not in string.punctuation]
       review=[pos_stem.stem(word) for word in review]
       review=" ".join(review)
       return(review)
    Input=clean(input_text)
    final_input=cv.transform([Input])
    Model=joblib.load("spam_mail&messages")
    pred=Model.predict(final_input)
    if pred==0:
        st.header("Not a spam")
    elif pred==1:
        st.header("Spam")


    



