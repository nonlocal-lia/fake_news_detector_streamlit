print('Importing Necessary Packages...')
from re import M
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import pickle
print('Packages Imported')

print('Loading Data...')
data = pd.read_csv('../data/clean_data.zip')
label = LabelEncoder()
X = data['no_stop_text']
y = label.fit_transform(data['label'])
print('Data Loaded')

print('Fitting Vectors...')
tfidf = TfidfVectorizer(max_features=1000)
X_train_vectorized = tfidf.fit_transform(X)
print('Vectors Fit')

print('Training Model...')
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_vectorized, y)
model = make_pipeline(tfidf, logistic_model)
print('Model Trained')

with open('./label_encoder.pickle', 'wb') as f:
    pickle.dump(label, f)
with open('./logistic_model.pickle', 'wb') as f:
    pickle.dump(model, f)
print('Model Saved!')