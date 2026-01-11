import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

def forecast_next(history, steps=1):
    """
    Forecast next risk value using ARIMA if enough history,
    otherwise fallback to simple Linear Regression.
    Returns (pred_mean, [lower, upper]) or (None, None).
    """
    try:
        hist_array = np.array(history, dtype=float)
        n = len(hist_array)

        if n < 2:
            return None, None  # not enough even for linear

        # ✅ Use ARIMA if we have enough history
        if n >= 10:
            try:
                model = ARIMA(hist_array, order=(2, 1, 1))
                fitted = model.fit()
                forecast = fitted.get_forecast(steps=steps)
                pred_mean = float(forecast.predicted_mean[0])
                conf_int = forecast.conf_int()[0].tolist()
                return round(pred_mean, 3), [round(c, 3) for c in conf_int]
            except Exception as e:
                print(f"[ARIMA Fallback Error] {e}")

        # ✅ Fallback: Linear Regression trend
        X = np.arange(n).reshape(-1, 1)
        y = hist_array
        model = LinearRegression()
        model.fit(X, y)
        next_x = np.array([[n]])
        pred = model.predict(next_x)[0]

        # crude "confidence interval"
        ci = [pred - np.std(y), pred + np.std(y)]
        return round(float(pred), 3), [round(float(ci[0]), 3), round(float(ci[1]), 3)]

    except Exception as e:
        print(f"[Forecast Error] {e}")
        return None, None
