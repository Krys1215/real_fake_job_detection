import re
import spacy
import os
import time
import pickle
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences

def data_cleanups(df: pd.DataFrame):
    #Data Cleanups

    df['text']=df['text'].str.replace('\n','')
    df['text']=df['text'].str.replace('\r','')
    df['text']=df['text'].str.replace('\t','')
  
  #This removes unwanted texts
    df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
  
  #Converting all upper case to lower case
    df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)
  
  #Remove un necessary white space
    df['text']=df['text'].str.replace('  ',' ')

  #Remove Stop words
    nlp=spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm 
    df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
    
    sp = spacy.load('en_core_web_sm')
    output=[]

    for sentence in df['text']:
        sentence=sp(str(sentence))
        s=[token.lemma_ for token in sentence]
        output.append(' '.join(s))
    df['processed']=pd.Series(output)
    
    return df


def read_tokenizer():
    script_dir = os.path.dirname(__file__)  # Get the directory of the script
    file_path = os.path.join(script_dir, 'tokenizer.pickle')

    with open(file_path, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    return loaded_tokenizer
  
def read_rf():
    script_dir = os.path.dirname(__file__)  # Get the directory of the script
    file_path = os.path.join(script_dir, 'random_forest.pickle')
    
    with open(file_path, 'rb') as handle:
        model = pickle.load(handle)
        
    return model
  
def read_lr():
    script_dir = os.path.dirname(__file__)  # Get the directory of the script
    file_path = os.path.join(script_dir, 'logistic_regression.pickle')
    
    with open(file_path, 'rb') as handle:
        model = pickle.load(handle)
        
    return model
  
def read_xgb():
    script_dir = os.path.dirname(__file__)  # Get the directory of the script
    file_path = os.path.join(script_dir, 'xgboost.pickle')
    
    with open(file_path, 'rb') as handle:
        model = pickle.load(handle)
        
    return model

def main():
    st.write("""
    Insert the job description to predict:
             """)
    text_to_predict = st.text_area("Text to predict")
    
    predict_data = [text_to_predict] 

    df_predict = pd.DataFrame({'text' : predict_data})
    
    # read the tokenizer here
    tokenizer = read_tokenizer()
    # data preprocessing here
    ready_to_predict = data_cleanups(df_predict)
    
    values_to_predict = tokenizer.texts_to_sequences(ready_to_predict['processed'].values)    # Tokenize the dataset
    values_to_predict = pad_sequences(values_to_predict, maxlen=250)  
    
    with st.spinner('Processing the data...'):
      time.sleep(1)
    st.write("After data processing:")
    
    st.dataframe(ready_to_predict['processed'])
    
    st.write("After tokenized: ")
    st.dataframe(values_to_predict)
    
    # read our models
    model_rf = read_rf()
    # model_lr = read_lr()
    # model_xgb = read_xgb()
    # then do the prediction
    rf_predict = model_rf.predict(values_to_predict)[0]
    # lr_predict = model_lr.predict(values_to_predict)[0]
    # xgb_predict = model_xgb.predict(values_to_predict)[0]
    # show the prediction
    with st.spinner('Predicting...'):
      time.sleep(1)
    st.success(f"The given job desciption is more likely to be: {'Fake' if rf_predict == 1 else 'Real'}")
    #st.warning(f"Logistic Regression: {lr_predict}")
    #st.info(f"XGBoost: {xgb_predict}")
    
    
if __name__ == '__main__':
    main()
    