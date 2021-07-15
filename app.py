import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import csv
from io import BytesIO
import base64

import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from flask import *
import json

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload():
    targetCol=request.form.get('target')
    select=request.form.get('dataType')

    file=request.files['file-upload-input']
    file.save(file.filename)
    filename=file.filename

    if not targetCol or not file:
        return render_template('index.html', error="Enter correct data")

    elif select=="numeric":
        d=dict()
        df=pd.read_csv(filename)
        targetCol=int(targetCol)
        d["shape"]=getShape(df,targetCol)
        d["countUniqueCategories"]=getCountUniqueCategories(df,targetCol)
        d["sourceFrequency"]=getSourceFrequency(df,targetCol)
        d["targetSourceCorr"]=getTargetSourceCorr(df,targetCol)
        d["accuracyTable"]=mlCompare(df,targetCol)
        return render_template('visual.html',data=d)

    else:
        d=dict()
        df=pd.read_csv(filename)
        targetCol=int(targetCol)
        d["shape"]=getShape(df,targetCol)
        d["countUniqueCategories"]=getCountUniqueCategories(df,targetCol)
        d["classFrequency"]=getClassFrequency(df,targetCol)
        d["commonWords"]=getCommonWords(df,targetCol)
        d["f1scoreandaccuracyTable"]=mlCompare_text(df,targetCol)
        return render_template('visual_text.html',data=d)

#text
def getClassFrequency(df,targetCol):
    return df[df.columns[targetCol]].value_counts().sort_index().to_frame().to_html()

def getCommonWords(df,targetCol):
    from collections import Counter
    from sklearn.feature_extraction import text
    stop = text.ENGLISH_STOP_WORDS
    stop=list(stop)
    symbols="{}()[].,:;+-*/&|<>=~$1234567890"
    for c in symbols:
        stop.append(c)

    imgs=[]
    for specialty in df[df.columns[targetCol]].unique():
        df2=df[df[df.columns[targetCol]]==specialty]
        sourceCol=0
        if targetCol==0:
            sourceCol=1
        df2[df2.columns[sourceCol]] = df2[df2.columns[sourceCol]].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))
        counterList=Counter(" ".join(df2[df2.columns[sourceCol]]).split()).most_common(10)
        word=[]
        count=[]
        for word_count in counterList:
            word.append(word_count[0])
            count.append(word_count[1])
        
        plt.figure()    
        plt.style.use('ggplot')
        plt.barh(word,count)
        plt.title(specialty)
        plt.ylabel('Word')
        plt.xlabel('Frequency')
        img = BytesIO()
        plt.savefig(img,format='png')
        plt.close()
        img.seek(0)

        imgs.append(base64.b64encode(img.getvalue()).decode('utf8'))
    
    return imgs

def mlCompare_text(df,targetCol):
    sourceCol=0
    if targetCol==0:
        sourceCol=1

    df.drop(df.columns.difference([df.columns[sourceCol],df.columns[targetCol]]), 1, inplace=True)

    #removing classes with less than 100
    s = df[df.columns[targetCol]].value_counts()
    df=df[df.isin(s.index[s >= 100]).values]

    source=df[df.columns[sourceCol]]
    target=df[df.columns[targetCol]]

    labels=target.unique().tolist()
    
    from sklearn.feature_extraction import text
    stop = text.ENGLISH_STOP_WORDS
    stop=list(stop)
    symbols="{}()[].,:;+-*/&|<>=~$1234567890"
    for c in symbols:
        stop.append(c)
        
    df[df.columns[sourceCol]] = df[df.columns[sourceCol]].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))

    from sklearn.model_selection import train_test_split
    test_percent=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    clfs=[ 
            Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())]),
            Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier())]),
            Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(max_depth=10))])
        ]
    values=[["NB"],["SGD"],["RF"]]

    import numpy as np
    from sklearn.metrics import f1_score

    f1scoretables=[]
    for p in test_percent:
        source_train, source_test, target_train, target_test=train_test_split(source,target,test_size=p,random_state=42)
        r=0
        for clf in clfs:
            clf.fit(source_train,target_train)
            target_pred=clf.predict(source_test)
            score=np.mean(target_pred == target_test)
            values[r].append(score)
            
            print(values[r][0], "Train size "+str(math.ceil((1-p)*100))+"%")
            score=f1_score(target_test,target_pred,zero_division=1,labels=labels,average=None)
            score=np.round(score,2)
            score=score.tolist()
            
            from prettytable import PrettyTable
            mytable=PrettyTable()
            mytable.title=str(values[r][0]+" "+"Train size "+str(math.ceil((1-p)*100))+"%" )
            mytable.field_names=["Class","f1 score"]
            for i in range(len(score)):
                mytable.add_row([labels[i],score[i]])
            f1scoretables.append(mytable.get_html_string())
            
            r+=1

    header=["Classifier"]
    for p in test_percent:
        header.append("Train size "+str(math.ceil((1-p)*100))+"%")

        
    accuracy_table=[]
    accuracy_table.append(header)
    for v in values:
        accuracy_table.append(v)

    import pandas

    accuracy_table=pandas.DataFrame(accuracy_table)

    return [f1scoretables,accuracy_table.to_html()]

##numeric
def getShape(df,targetCol):
    return df.shape

def getCountUniqueCategories(df,targetCol):
    return len(df[df.columns[targetCol]].unique())

def getSourceFrequency(df,targetCol):
    source_freq=[]
    for column in df:
        if column!=df.columns[targetCol]:
            source_freq.append( [column,len( df[df[column]==1] )] )
    source_freq.sort(key=lambda x:x[1])

    source=[]
    freq=[]
    for data in source_freq:
        source.append(data[0])
        freq.append(data[1])

    plt.figure(figsize=(30, 30), dpi=80)    
    plt.style.use('ggplot')
        
    plt.barh(source,freq)
    plt.title('Source Frequency')
    plt.ylabel('Source')
    plt.xlabel('Frequency')
    
    plt.plot()
    img = BytesIO()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode('utf8')

def getTargetSourceCorr(df,targetCol):

    imgs=[]
    for id in df[df.columns[targetCol]].unique(): 
        corr=df.corrwith(df[df.columns[targetCol]]==id)
        corr=corr[corr>0]
        corr=corr.to_frame()
        sns.heatmap(corr,annot=True)

        plt.plot()
        img = BytesIO()
        plt.savefig(img,format='png')
        plt.close()
        img.seek(0)

        imgs.append(base64.b64encode(img.getvalue()).decode('utf8'))
    
    return imgs

def mlCompare(df,targetCol):
    data=df.to_numpy()
    target=data[:,targetCol:targetCol+1]
    np.delete(data,targetCol,1)
    source=data

    from sklearn.model_selection import train_test_split
    test_percent=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    clfs=[DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=3),LinearDiscriminantAnalysis(),GaussianNB(),RandomForestClassifier(),SGDClassifier()]
    values=[["DTC"],["KNN"],["LDA"],["NB"],["RF"],["SGD"]]

    from sklearn.metrics import accuracy_score

    for p in test_percent:
        source_train, source_test, target_train, target_test=train_test_split(source,target,test_size=p,random_state=42)
        r=0
        for clf in clfs:
            clf.fit(source_train,target_train.ravel())
            target_pred=clf.predict(source_test)
            score=accuracy_score(target_test, target_pred)
            values[r].append(score)
            r+=1
    
    import math

    header=["Classifier"]
    for p in test_percent:
        header.append("Train size "+str(math.ceil((1-p)*100))+"%")
        
    accuracy_table=[]
    accuracy_table.append(header)
    for v in values:
        accuracy_table.append(v)

    import pandas

    accuracy_table=pandas.DataFrame(accuracy_table)
    
    return accuracy_table.to_html()
    