import joblib
from joblib import load
import pandas as pd
import numpy as np

# load the mode
model = "ml_scripts/model/model.joblib"
load_model = joblib.load(model)


def runModel():
    # load the csv
    test = pd.read_csv('instance/testCSV/test.csv', sep=',')

    X_to_push = test
    X_testing = test.drop(['Name'], axis=1)

    clf = load_model
    X_testing_scaled = clf.named_steps['scale'].transform(X_testing)
    X_testing_pca = clf.named_steps['pca'].transform(X_testing_scaled)
    y_testing_pred = clf.named_steps['clf'].predict_proba(X_testing_pca)
    df = pd.concat([X_to_push['Name'], pd.DataFrame(y_testing_pred)], axis=1)

    print(df)
    return (df)
