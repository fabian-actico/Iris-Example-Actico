import pandas as pd
import sklearn
import pickle

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def train_model():

    iris = datasets.load_iris()
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],'species': iris.target})
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    file_to_store = open('model.pickle', "wb")
    pickle.dump(model, file_to_store)
    file_to_store.close()

if __name__ == '__main__':
    train_model()

