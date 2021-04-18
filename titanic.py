from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def load_data(path):
    df = pd.read_csv(path)

    return df


def split_feature_label(Xy):
    y = Xy['Survived']
    X = Xy.drop('Survived', axis=1, inplace=False)
    return X, y


def transform(X):
    X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    X['Age'].fillna(X['Age'].mean(), inplace=True)
    X['Embarked'].fillna('N', inplace=True)
    X['Fare'].fillna(X['Fare'].mean(), inplace=True)

    string_features = ['Sex', 'Embarked']
    for feature in string_features:
        label_encoder = LabelEncoder()
        X[feature] = label_encoder.fit_transform(X[feature])

    return X


def train_model(X, y):
    param_grid = {
        'max_depth': [2, 3, 5, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 5, 8]
    }
    decision_tree_classifier = DecisionTreeClassifier()
    grid_search_cv = GridSearchCV(decision_tree_classifier, param_grid=param_grid, scoring='accuracy', cv=5)

    grid_search_cv.fit(X, y)

    return grid_search_cv.best_estimator_


def parse_args():
    parser = ArgumentParser(
        description='Generate the submission file for Kaggle Titanic competition.')
    parser.add_argument(
        '--train', type=Path, default='train.csv',
        help='path of train.csv downloaded from the competition')
    parser.add_argument(
        '--test', type=Path, default='test.csv',
        help='path of test.csv downloaded from the competition')

    return parser.parse_args()


def main(args):
    Xy_train = load_data(args.train)
    X_train, y_train = split_feature_label(Xy_train)
    X_train = transform(X_train)
    model = train_model(X_train, y_train)

    X_test = load_data(args.test)
    passenger_ids = X_test['PassengerId']
    X_test = transform(X_test)
    y_test = model.predict(X_test)

    submission = {
        'PassengerId': passenger_ids,
        'Survived': y_test
    }
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
