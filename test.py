from random import shuffle
from pickle import load as load_p
from joblib import load as load_j
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI
from numpy import argsort,sort
from gc import collect
app = FastAPI()
def lem(query,lemmatizer):
    return lemmatizer.lemmatize(query)

@app.get('/')
def index():
    return {'Message': 'Welcome to Chat Model'}

@app.post('/predict_text')
def predict_text(qu : str):
    if qu[-1] == "#":
        MLP = load_p(open('mlp.sav', 'rb'))
        with open('symp.pkl', 'rb') as f:
            symp = load_p(f)
        with open('dis.pkl', 'rb') as f:
            dis = load_p(f)
        with open('symptoms4.pkl', 'rb') as f:
            symptoms4 = load_p(f)
        qu = qu[:len(qu) -1]
        qu = qu.split("#")
        #res = diseasePrediction2(qu,MLP,symp,symptoms4,dis)
        del qu , MLP,symp,symptoms4,dis
        collect
        return {'prediction':'res+"2"'}
    else:
        lemmatizer = WordNetLemmatizer()
        qu = lem(qu,lemmatizer)
        Pipe2=load_j('withoutTreatment_withSymp2.pkl')
        ans=Pipe2.predict([qu])[0]
        del qu,lemmatizer,Pipe2
        collect()
        ans=ans.replace('_',' ')
        if ans in symp:
            return{'prediction':ans+"1"}
        else:
            return {'prediction':ans+"0"}