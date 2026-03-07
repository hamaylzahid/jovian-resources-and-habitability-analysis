from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_classification_models(X_train,y_train):

    logistic_model = LogisticRegression()

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=78,
        random_state=42
    )

    logistic_model.fit(X_train,y_train)

    rf_model.fit(X_train,y_train)

    return logistic_model,rf_model
