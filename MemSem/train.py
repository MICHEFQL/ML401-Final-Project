import pickle
import os
import pandas as pd
import numpy as np
from model import Classifier
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
random.seed(42)
cls = Classifier(epochs=1, batch_size=100, metrics=True, plot_model_diagram=True, summary=True)

# with open("data.pkl","rb") as pickle_in:
#   data = pickle.load(pickle_in)
# pickle_in.close()
# print(data)
# cls.train(data)

image_folder = '/Users/poppy_puppet/Downloads/cat-memes-data-v-anzi'
csv_path = 'cat_memes - Sheet1.csv'
output_base = './dataset'

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df['image'] = df['image'].str.strip()
df['label'] = df['label'].str.strip().str.lower()

df['image'] = df['image'].apply(lambda x: os.path.join(image_folder, x))

X = df.drop('label', axis=1)  # Features
y = df['label']               # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)


cls.train(train_df)
# print(df['image'][1])
# e = cls.evaluate(df)
# print(e)



pairs = []
for index, row in test_df.iterrows():
    x = row['image']
    y = row['label']
    pairs.append([y,cls.predict(x)])
pairs = np.array(pairs)
print(pairs)
y_true = pairs[:,0]
y_pred = pairs[:,1]
p = precision_score(y_true, y_pred , average='macro')
a = accuracy_score(y_true, y_pred)
r = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print(" ")
print("Test set evaluation: ")
print("precision  "+str(p))
print("recall  "+str(r))
print("f1  "+str(f1))
print("accuracy "+str(a))

# print(test_pred)
# print(test_df['label'])
# print(cls.evaluate(test_df))
# print("done training")

