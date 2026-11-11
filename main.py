import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("train.txt",sep=";",header=None,names=["text","emotions"])
unique_emotions=df["emotions"].unique()
emotion_number={}
i=0
for emo in unique_emotions:
    emotion_number[emo]=i
    i+=1
print(emotion_number)    
    
df["emotions"]=df["emotions"].map(emotion_number)
print(df.head())

df['text']=df['text'].apply(lambda x: x.lower())
#bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
x=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, df['emotions'], test_size=0.2, random_state=42) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 

#naive bayes
from sklearn.naive_bayes import MultinomialNB 
model1=MultinomialNB()
model1.fit(X_train, y_train) 
y_pred=model1.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#prediction for a sample text
y_pred = model.predict(vectorizer.transform(["I am so happy today!"]))
y_pred1=model1.predict(vectorizer.transform(["I am so happy today!"]))
print("logistic regression prediction:", y_pred)
print("naive bayes prediction:", y_pred1)