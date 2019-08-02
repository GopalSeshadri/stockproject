import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_html_components as html
import dash_core_components as core
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import pandas_datareader as web
import calendar
import datetime
from pmdarima.arima import auto_arima

app = dash.Dash()
now = dt.now()
DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

nasdaq_df = pd.read_csv("Data/NASDAQcompanylist.csv")
symbols_options = [{'label' : nasdaq_df[nasdaq_df['Symbol'] == Symbol]['Name'] + ' [{}]'.format(Symbol), 'value' : Symbol } for Symbol in nasdaq_df['Symbol'].unique()]

# Creating Application Layout
app.layout = html.Div([
    html.Div([
        html.H1('Stock Price Comparer')
    ], style =  dict(width = '100%',
                height = '5%',
                display = 'flex')),
    html.Div([
        html.Div([
            html.H3('Select Stock code', style = dict(height = '30%')),
            core.Dropdown(id = 'stock-code-input',
                    options = symbols_options,
                    value = 'TSLA',
                    multi = True,
                    style = dict(height = '50%',
                                fontSize = 16))
        ], style = dict(width = '30%',
                        float = 'left',
                        display = 'inline-block')),

        html.Div([
            html.H3('Select Date Range', style = dict(height = '30%')),
            core.DatePickerRange(
            id ='date-picker-range',
            start_date = dt(now.year - 1, now.month, now.day),
            end_date = now,
            style = dict(height = '50%'))
        ], style = dict(width = '24%',
                        float = 'left',
                        display = 'inline-block',
                        marginLeft = '30px')),

        html.Div([
            html.H3('', style = dict(height = '30%')),
            html.Button(id = 'submit-button',
                    n_clicks = 0,
                    children = 'Submit',
                    style = dict(height = '50%',
                                fontSize = 24))
        ], style = dict(width = '10%',
                        float = 'left',
                        display = 'inline-block'))

    ], style =  dict(width = '100%',
                    height = '15%',
                    display = 'flex')),
    html.Div([
        core.Loading(id = 'stock-loading', children = [
                core.Graph(id = 'stock-price-graph',
                        style = dict(width = '200%'))],
                type = 'default',
                style =  dict(width = '200%'))


    ], style =  dict(width = '100%',
                    height = '50%',
                    display = 'flex',
                    marginTop = '30px'))
], style = dict(fontFamily = 'Helvetica',
                width = '100%'))

@app.callback(Output(component_id = 'stock-price-graph', component_property = 'figure'),
        [Input(component_id = 'submit-button', component_property = 'n_clicks')],
        [State(component_id = 'stock-code-input', component_property = 'value'),
        State(component_id = 'date-picker-range', component_property = 'start_date'),
        State(component_id = 'date-picker-range', component_property = 'end_date')])
def affect_stockpricegraph(nclicks, stockcode, startdate, enddate):
    start = dt.strptime(startdate[:10], '%Y-%m-%d')
    end = dt.strptime(enddate[:10], '%Y-%m-%d')
    if type(stockcode) != list:
        stockcode = [stockcode]

    data_df = web.get_data_yahoo(stockcode, start, end)
    data_df.index = pd.to_datetime(data_df.index)
    end_year = end.year
    end_month = end.month
    end_day = end.day


    date_list = [end + datetime.timedelta(days = x) for x in range(0, 60)]

    data_df = [data_df['Close', sc] for sc in stockcode]

    stepwise_model = [auto_arima(data_df[each], start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True) for each in range(len(stockcode))]

    # print(stepwise_model.aic())
    data_df = pd.concat(data_df, axis = 1)

    data_df.columns = [sc for sc in stockcode]

    future_forecast = [stepwise_model[each].predict(n_periods = 60) for each in range(len(stockcode))]
    forecast_dict = {'Date' : [dt.strptime(str(each.date())[:10], '%Y-%m-%d') for each in date_list]}
    for each in range(len(stockcode)):
        forecast_dict[stockcode[each]] = future_forecast[each]

    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df['Class'] = 'Predicted'
    forecast_df = forecast_df.set_index(['Date'])

    data_df['Class'] = 'Actual'

    final_df = pd.concat([data_df, forecast_df])

    #print(data_df)

    data = [go.Scatter(x = final_df[final_df['Class'] == 'Actual'].index,
                    y = np.ravel(final_df[final_df['Class'] == 'Actual'][stockcode[each]].values),
                    mode = 'lines',
                    name = '{}-Actual'.format(stockcode[each]),
                    line = dict(dash = 'solid',
                            color = DEFAULT_PLOTLY_COLORS[each])) for each in range(len(stockcode))]

    preddata = [go.Scatter(x = final_df[final_df['Class'] == 'Predicted'].index,
                    y = np.ravel(final_df[final_df['Class'] == 'Predicted'][stockcode[each]].values),
                    mode = 'lines',
                    name = '{}-Predicted'.format(stockcode[each]),
                    line = dict(dash = 'dash',
                            color = DEFAULT_PLOTLY_COLORS[each])) for each in range(len(stockcode))]

    for each in preddata:
        data.append(each)

    layout = go.Layout(title = dict(text = 'Stock Price of {}'.format(stockcode),
                            font = dict(size = 24)),
                        yaxis = dict(title = dict(text = 'Stock Price in USD')))

    fig = dict(data = data, layout = layout)
    return fig

if __name__ == '__main__':
    app.run_server()
