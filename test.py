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
def getRes(model,psymptoms,symp):
    l2=[]
    for i in range(0,len(symp)):
        l2.append(0)
    for k in range(0,len(symp)):
        for z in psymptoms:
            if(z==symp[k]):
                l2[k]=1
    inputtest = [l2]
    predict = model.predict(inputtest)
    predicted=predict[0]
    probs = model.predict_proba(inputtest)
    best_n = argsort(-probs, axis=1)
    best_n2 = -1*sort(-probs, axis=1)
    newDis = []
    for i in range(len(best_n[0])):
        x=round(best_n2[0][i]*100)
        if(x>=1):
            newDis.append(best_n[0][i])

    return newDis
def diseasePrediction2(query,MLP,symp,symptoms4,dis):
    allS=[]
    allS2=[]
    indx=0
    res=""
    newDis = getRes(MLP,query,symp)
    m = 0
    for i in query:
        if i in symptoms4[newDis[0]]:
            m=m+1
        acc=(m/len(symptoms4[newDis[0]]))
        if acc*100 >=70:            
            res2={}
            for i in range(len(newDis)):
                m=0
                for j in query:
                    if j in symptoms4[newDis[i]]:
                        m=m+1
                acc=(m/len(symptoms4[newDis[i]]))
                res2[newDis[i]]=acc
            keys= sorted(res2, key=res2.get, reverse=True)[:4]
            return 'you are suffer from ' + dis[keys[0]] + 'with acc ' + str(res2[keys[0]]) + "@" + 'you are suffer from ' + dis[keys[1]] + 'with acc ' + str(res2[keys[1]])+ "@"+'you are suffer from ' + dis[keys[2]] + 'with acc ' + str(res2[keys[2]])+ "@"+'you are suffer from ' + dis[keys[3]] + 'with acc ' + str(res2[keys[3]])
    for i in newDis:
        if query[-1] in symptoms4[i]:
            allS.append(symptoms4[i])
    for i in range(len(allS)):
        for j in range(len(allS[i])):
            allS2.append(allS[i][j].lower())
    shuffle(allS2)    
    if len(allS2)==0:
        return "i cant't detect what you are sufferd from so please try to go to real doctor :)"
    else:
        return '#'.join(query) + '$' + str(allS2)

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
        res = diseasePrediction2(qu,MLP,symp,symptoms4,dis)
        del qu , MLP,symp,symptoms4,dis
        collect
        return {'prediction':res+"2"}
    else:
        lemmatizer = WordNetLemmatizer()
        qu = lem(qu,lemmatizer)
        Pipe2=load_j('withoutTreatment_withSymp3.pkl')
        ans=Pipe2.predict([qu])[0]
        del qu,lemmatizer,Pipe2
        collect()
        ans=ans.replace('_',' ')
        if ans in symp:
            return{'prediction':ans+"1"}
        else:
            return {'prediction':ans+"0"}