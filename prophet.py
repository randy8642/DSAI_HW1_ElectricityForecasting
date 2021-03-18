'''
https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''

from fbprophet import Prophet

def forecastByProphet(trainData, predictNum):

    m = Prophet()
    m.fit(trainData)

    future = m.make_future_dataframe(periods=predictNum)
    forecast = m.predict(future)

    output = forecast[['yhat']][-predictNum:].to_numpy().flatten()

    return output
