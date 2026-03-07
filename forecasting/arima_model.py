
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def apply_arima_forecasting(data):
    data['ARIMA Forecasted Years'] = np.nan
    for index in range(len(data)):
        try:
            history = data.loc[:index,'Years of Sustainable Extraction'].dropna().values[-5:]
            if len(history) < 5:
                data.at[index,'ARIMA Forecasted Years'] = data.at[index,'Years of Sustainable Extraction']
            else:
                model = ARIMA(history, order=(1,1,0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)[0]
                data.at[index,'ARIMA Forecasted Years'] = max(forecast,0)
        except Exception:
            data.at[index,'ARIMA Forecasted Years'] = data.at[index,'Years of Sustainable Extraction']
    return data
