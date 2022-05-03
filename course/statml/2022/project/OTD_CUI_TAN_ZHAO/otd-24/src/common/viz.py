from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy

def do_classification(x, y, x_test, y_test, bb="rf", __rf_n_estimators=100,__ada_n_estimators=100):
    # just trains a classifiers on provided data
    # if simple ... train a RF classfier
    # RF is fast and so useful when training a classfier during validation
    # expects numpy array as input

    # do scaling
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    if bb=='rf':
        # this should be faster and better than training a NN
        model = RandomForestClassifier(n_jobs=-1, n_estimators=__rf_n_estimators)
    elif bb=='ada':
        model = AdaBoostClassifier(n_estimators=__rf_n_estimators)
    elif bb=='lr':
        model = LogisticRegression(max_iter=1000)
    else:

        model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, n_iter_no_change=10)

    y = y.ravel()
    y_test = y_test.ravel()

    model.fit(x, y)
    score = model.score(x_test, y_test)
    prob = model.predict_proba(x_test)
    prob_labels=numpy.argmax(prob, axis=1)
    f1=f1_score(y_test,prob_labels,average='macro')

    return score,f1, prob
