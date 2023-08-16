import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import time
from fastapi import FastAPI
import pickle
import uvicorn
from pydantic import BaseModel
import json

pre_defined_corpus = {'text': [
    'I really like to eat ice cream',
    'I am a student',
    'Hello I am a student',
    'I like being a student',
    'I am an old student',
    'thank you for the ice cream',
    'I like your name',
    'Thank you very much',
    'I am 5 years old',
    '5 ice cream please',
    'I like to eat 5 ice cream'
]}

def tokenisasi(text):
    return text.split()

def get_embeddings(model):
    embedding = []

    for line in df['clean']:
        word_embedding = None
        count = 0
        for word in line:
            if word in model.wv.key_to_index:
                count += 1
                if word_embedding is None:
                    word_embedding = model.wv[word]
                else:
                    word_embedding = word_embedding + model.wv[word]
                
        if word_embedding is not None:
            word_embedding = word_embedding / count
            embedding.append(word_embedding)
    
    return embedding

def get_target_embedding(target):
    count = 0
    embedding2 = []
    target_words_embedding = None

    for text in target:
        if text in model1.wv.key_to_index:
            count  += 1
            if target_words_embedding is None:
                target_words_embedding = model1.wv[text]
            else:
                target_words_embedding = target_words_embedding +model1.wv[text]

    if target_words_embedding is not None:
        target_words_embedding = target_words_embedding / count
    
    embedding2.append(target_words_embedding)

    return embedding2

def get_recommendation(embedding, target):
    cosine_similarities = cosine_similarity(embedding, target)

    sim_scores = list(enumerate(cosine_similarities))
    highest_score = 0
    idx = 0

    for i in range(0,8):
        if sim_scores[i][1][0] > highest_score:
            highest_score = sim_scores[i][1][0]
            idx = sim_scores[i][0]

    print(highest_score)
    print(idx)

    return df['text'][idx]

     

df = pd.DataFrame(pre_defined_corpus)
df['clean'] = df['text'].apply(tokenisasi)

with open('model1.pkl', 'rb') as f:
        model1 = pickle.load(f)

embedding = get_embeddings(model1)

class Item(BaseModel):
    words: list = []

app = FastAPI()

@app.post("/")
async def root(item: Item):
    list_words = []
    for w in item.words:
        list_words.append(w)
    
    print('list words:', list_words)

    # target_words = ['i', 'hungry']

    em = get_target_embedding(list_words)

    recommendation = get_recommendation(embedding, em)
    print('recommendation: ', recommendation)

    return {'words recommendation': recommendation}
    

if __name__ == '__main__':
    uvicorn.run(app)

    


