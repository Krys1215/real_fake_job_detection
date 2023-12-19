import re
import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
import pickle

#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spacy.lang.en import English

def read_tokenizer_wordIndex():
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    # 加载 word_index
    with open('word_index.pickle', 'rb') as handle:
        loaded_word_index = pickle.load(handle)
        
    return loaded_tokenizer, loaded_word_index
    
    
def what_if(df: pd.DataFrame, tokenizer, word_index):
    nlp = spacy.load('en_core_web_sm')
    
    # Data cleaning
    df['text']=df['text'].str.replace('\n','')
    df['text']=df['text'].str.replace('\r','')
    df['text']=df['text'].str.replace('\t','')
  
    # This removes unwanted texts
    df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
  
    # Converting all upper case to lower case
    df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)
  
    # Remove un necessary white space
    df['text']=df['text'].str.replace('  ',' ')


    df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))
    
    # Lemmanization
    sp = spacy.load('en_core_web_sm')
    output=[]
    for sentence in df['text']:
        sentence=sp(str(sentence))
        s=[token.lemma_ for token in sentence]
        output.append(' '.join(s))
        df['processed']=pd.Series(output)
    
    # Tokenization and padding
    vocab_size = 100000
    embedding_dim = 64
    max_length = 250
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000
    X_sequences = tokenizer.texts_to_sequences(df['processed'].values)  
    
    # 使用word_index将单词映射为整数
    X = []
    for seq in X_sequences:
        int_seq = [word_index[word] for word in seq if word in word_index]
        X.append(int_seq)
    
    # 填充序列
    X_padded = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')
    
    return X_padded 
    

def preprocess_new_data(new_data, loaded_tokenizer, loaded_word_index):
    sp = spacy.load('en_core_web_sm')
    new_data['text'] = new_data['text'].apply(lambda x: re.sub(r'[0-9]','',x))
    new_data['text'] = new_data['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
    new_data['text'] = new_data['text'].apply(lambda s:s.lower() if type(s) == str else s)
    new_data['text'] = new_data['text'].str.replace('  ',' ')
    new_data['text'] = new_data['text'].apply(lambda x: ' '.join([word for word in x.split() if loaded_word_index.get(word, 0) not in loaded_tokenizer.index_word]))
    # Lemmatization
    output = []
    for sentence in new_data['text']:
        sentence = sp(str(sentence))
        s = [token.lemma_ for token in sentence]
        output.append(' '.join(s))
    new_data['processed'] = pd.Series(output)

    # Tokenization and padding
    
    max_length = 250
    X_new = loaded_tokenizer.texts_to_sequences(new_data['processed'].values)
    X_new = pad_sequences(X_new, maxlen=max_length) # Padding the dataset
    return X_new

    

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

