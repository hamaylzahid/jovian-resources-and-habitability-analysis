from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def regression_metrics(y_true,y_pred):

    print("MAE:",mean_absolute_error(y_true,y_pred))
    print("MSE:",mean_squared_error(y_true,y_pred))
    print("R2:",r2_score(y_true,y_pred))


def classification_metrics(y_true,y_pred):

    print("Accuracy:",accuracy_score(y_true,y_pred))
    print("Precision:",precision_score(y_true,y_pred))
    print("Recall:",recall_score(y_true,y_pred))
    print("F1:",f1_score(y_true,y_pred))
