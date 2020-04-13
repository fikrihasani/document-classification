from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def preprocessing(data, feature=True):
    if feature:
        import string
        exclude = set(string.punctuation)
        try:
            x = ''.join(ch for ch in data if ch not in exclude)
        except:
            pass
        return x
    else:
        labelencoder = LabelEncoder()
        return labelencoder.fit_transform(data)


def feature_extraction(docs, feature_type):
    if feature_type == "tf-idf" or feature_type == "tfidf":
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(docs)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        return X_train_tfidf
    else:
        pass


def build(X, y):
    X = feature_extraction(X.apply(preprocessing), "tfidf")
    y = preprocessing(y, False)

    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # xgboost
    classifier = OneVsRestClassifier(XGBClassifier())
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=3))

    # svm
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model = SVC(kernel="rbf")
    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=3))

    # lstm

    # print('Best score for training data:', svm_model.best_score_,"\n")

    # View the best parameters for the model found using grid search
    # print('Best C:',svm_model.best_estimator_.C,"\n")
    # print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    # print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")
