from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_resource_model(data):

    features = [
        'Crust Stability Index',
        'Operational Risk Factor',
        'Replenishment Rate (tons/year)'
    ]

    X = data[features]
    y = data['Resource Availability (tons/km²)']

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=123
    )

    model = RandomForestRegressor(n_estimators=100,random_state=123)

    model.fit(X_train,y_train)

    data['Predicted Resource Availability (tons/km²)'] = model.predict(X)

    return model,data,X_test,y_test
