import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from io import BytesIO
import base64

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
    leftCol=request.form.get('left')
    rightCol=request.form.get('right')

    file=request.files['file-upload-input']
    file.save(file.filename)
    filename=file.filename

    if not targetCol or not file:
        return render_template('index.html', error="Enter correct data")
    else:
        d=dict()
        df=pd.read_csv(filename)
        targetCol=int(targetCol)
        d["shape"]=getShape(df,targetCol)
        d["countUniqueCategories"]=getCountUniqueCategories(df,targetCol)
        d["sourceFrequency"]=getSourceFrequency(df,targetCol)
        d["targetSourceCorr"]=getTargetSourceCorr(df,targetCol)
        return render_template('visual.html',data=d)


##visuals
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