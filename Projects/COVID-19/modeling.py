import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pmdarima.metrics import smape
from pmdarima.preprocessing import BoxCoxEndogTransformer
import matplotlib.pyplot as plt

def ARIMA(x, n_periods=14, normalize=False):

    # input a pandas Series
    # contain a time series data with timestamp as index

    # split data
    train = x[60:-n_periods]
    test = x[-n_periods:]

    # Box-Cox Transformation
    if normalize == True:
        boxcox = BoxCoxEndogTransformer(lmbda2=1e-6).fit(train)
        train, _ = boxcox.transform(train)
        test, _ = boxcox.transform(test)

    best_model = None
    best_scores = np.infty
        
    # Train SARIMA
    for i in range(1,3):
        for j in range(1,3):
            model = pm.auto_arima(train, m=7, max_p=3, max_q=3, max_P=3, max_Q=3, d=i, D=j, max_order=12,
                                  stepwise=True, out_of_sample_size=n_periods, scoring='mae', information_criterion='oob',
                                  error_action='ignore', trace=False, suppress_warnings=True)
            
            pred = model.predict(n_periods=n_periods)
            mae = mean_absolute_error(test, pred)
            if mae < best_scores:
                best_scores = mae
                best_model = model

    # Envaluation Metrics
    pred = best_model.predict(n_periods=n_periods)
    if normalize == True:
        pred, _ = boxcox.inverse_transform(pred)

    pred = pd.Series(pred, index=x.index[-n_periods:])
    r2 = round(r2_score(test, pred), 2)
    RMSE = round(np.sqrt(mean_squared_error(test, pred)), 2)
    MAE = round(mean_absolute_error(test, pred),2)
    SMAPE = round(smape(test, pred), 2)
    print('R2:', r2)
    print('RMSE is {}'.format(RMSE))
    print('MAE is {}'.format(MAE))
    print('SMAPE is {}'.format(SMAPE))
    
    ax = x.plot(label='Observed', figsize=(14, 4), linewidth=3)
    pred.plot(ax=ax, label='Forecasting', linewidth=3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()

    # Forecasting
    start = x.index[-1] + pd.Timedelta(1, unit='D')
    end = start + pd.Timedelta(n_periods - 1, unit='D')
    time_range = pd.date_range(start, end, freq='D')

    model.update(test)
    pred, confi = model.predict(n_periods=n_periods, return_conf_int=True)
    if normalize == True:
        pred, _ = boxcox.inverse_transform(pred)

    pred = pd.Series(pred, name='Forecasting', index=time_range).reset_index()
    confi = pd.DataFrame(confi, columns=['pred_lower', 'pred_upper'])

    pred['Order Date'] = pred['index'].dt.date.astype('datetime64[ns]')
    pred.set_index('Order Date', inplace=True)
    pred.drop('index', axis=1, inplace=True)
    
    # save results and plots
    pd.concat([pred, confi], axis=1).to_csv('forecasting.csv', index=False)
    ax = x.plot(label='Observed', figsize=(14, 4), linewidth=3)
    pred.plot(ax=ax, label='Forecasting', linewidth=3)
    ax.fill_between(pred.index,
                    confi.iloc[:, 0],
                    confi.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()
    
    return model
    
