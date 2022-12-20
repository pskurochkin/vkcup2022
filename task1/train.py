import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from fine_preprocessor import FinePreprocessorText


def main():
    train_df = pd.read_csv('task1/data/train.csv')

    # split data
    train_split, test_split = train_test_split(train_df, test_size=0.2, random_state=42)
    train_split = train_split.reset_index(drop=True)
    test_split = test_split.reset_index(drop=True)

    # preprocessing text
    preprocessor = FinePreprocessorText()
    train_split['text'] = train_split['text'].apply(preprocessor)
    test_split['text'] = test_split['text'].apply(preprocessor)

    # create vectorizer and encoder
    vectorizer = TfidfVectorizer(lowercase=False, max_features=10000)
    category_groups = train_split.groupby('category', as_index=False)
    corpus_df = category_groups.agg({'text': ' '.join}).reset_index(drop=True)
    vectorizer.fit(corpus_df.loc[:, 'text'])

    l_encoder = LabelEncoder()
    l_encoder.fit(train_split.loc[:, 'category'])

    # transform train and test data
    X_train = vectorizer.transform(train_split.loc[:, 'text'])
    y_train = l_encoder.transform(train_split.loc[:, 'category'])

    X_test = vectorizer.transform(test_split.loc[:, 'text'])
    y_test = l_encoder.transform(test_split.loc[:, 'category'])

    # fit and predict SVM classifier
    clf = SVC(probability=True, kernel='rbf')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cutom_metric =  2 * accuracy - 1
    print(f'accuracy: {accuracy:.5f}')
    print(f'+1/-1 metric: {cutom_metric:.5f}')

    # save predict for submission csv-format
    output_df = test_split.drop('text', axis=1)
    output_df['category'] = y_pred

    groupby = output_df.groupby('oid', as_index = False)
    output_df = groupby['category'].agg(lambda x: np.bincount(x).argmax())
    output_df['category'] = l_encoder.inverse_transform(output_df['category'])

    output_df.to_csv('task1/data/submission/tf-idf.csv', index=False)

if __name__ == '__main__':
    main()
