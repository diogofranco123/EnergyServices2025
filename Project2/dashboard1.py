import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load 2019 data
df_2019 = pd.read_csv('C:/Diogo/EnServ/Project2/testData_2019_Civil.csv')

df_2019["Date"] = pd.to_datetime(df_2019["Date"])
variables = df_2019.columns[1:]

# Load models
XGB_model = joblib.load('C:/Diogo/EnServ/Project2/XGB_model.pkl')
NN_model = joblib.load('C:/Diogo/EnServ/Project2/NN_model.pkl')
RF_model = joblib.load('C:/Diogo/EnServ/Project2/RF_model.pkl')
BT_model = joblib.load('C:/Diogo/EnServ/Project2/BT_model.pkl')

# Prepare features for model
df_new = df_2019.drop(columns=['windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'rain_mm/h', 'rain_day'])
df_new['Power-1'] = df_new['Civil (kWh)'].shift(1)
df_new = df_new.dropna()
df_new["hour"] = df_new['Date'].dt.hour
df_new["day_of_week"] = df_new['Date'].dt.dayofweek
df_new["hour_sin"] = np.sin(2 * np.pi * df_new["hour"] / 24)
df_new["hour_cos"] = np.cos(2 * np.pi * df_new["hour"] / 24)

df_new = df_new.iloc[:, [0,1,5,2,4,3,6,8,9,7]]

Z = df_new.values
Y = Z[:, 1]
X = Z[:, [2, 3, 4, 5, 6, 7, 8, 9]]
y_real_2019 = df_new['Civil (kWh)'].values

# XGB
y_pred_2019_XGB = XGB_model.predict(X)

MAE_XGB = metrics.mean_absolute_error(Y, y_pred_2019_XGB)
MBE_XGB = np.mean(Y - y_pred_2019_XGB)
MSE_XGB = metrics.mean_squared_error(Y, y_pred_2019_XGB)
RMSE_XGB = np.sqrt(MSE_XGB)
cvRMSE_XGB = RMSE_XGB / np.mean(Y)
NMBE_XGB = MBE_XGB / np.mean(Y)

# NN
y_pred_2019_NN = NN_model.predict(X)

MAE_NN = metrics.mean_absolute_error(Y, y_pred_2019_NN)
MBE_NN = np.mean(Y - y_pred_2019_NN)
MSE_NN = metrics.mean_squared_error(Y, y_pred_2019_NN)
RMSE_NN = np.sqrt(MSE_NN)
cvRMSE_NN = RMSE_NN / np.mean(Y)
NMBE_NN = MBE_NN / np.mean(Y)

# BT
y_pred_2019_BT = BT_model.predict(X)

MAE_BT = metrics.mean_absolute_error(Y, y_pred_2019_BT)
MBE_BT = np.mean(Y - y_pred_2019_BT)
MSE_BT = metrics.mean_squared_error(Y, y_pred_2019_BT)
RMSE_BT = np.sqrt(MSE_BT)
cvRMSE_BT = RMSE_BT / np.mean(Y)
NMBE_BT = MBE_BT / np.mean(Y)

# RF
y_pred_2019_RF = RF_model.predict(X)

MAE_RF = metrics.mean_absolute_error(Y, y_pred_2019_RF)
MBE_RF = np.mean(Y - y_pred_2019_RF)
MSE_RF = metrics.mean_squared_error(Y, y_pred_2019_RF)
RMSE_RF = np.sqrt(MSE_RF)
cvRMSE_RF = RMSE_RF / np.mean(Y)
NMBE_RF = MBE_RF / np.mean(Y)

# Dataframe with real data
df_real = pd.DataFrame({'Date': df_new['Date'].values, 'Real values': y_real_2019})

# Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1('IST Power Forecast tool CivilPav (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            # "IST Raw Data 2019"
            html.H4('IST Raw Data 2019'),
            dcc.Dropdown(id='variable-dropdown',
                         options=[{'label': var, 'value': var} for var in variables],
                         value=variables[0]),
            dcc.Graph(id='variable-graph'),

            # "Exploratory Analysis"
            html.H4('Exploratory Data Analysis'),
            dcc.Dropdown(id='eda-dropdown',
                         options=[
                             {'label': 'Boxplot', 'value': 'boxplot'},
                             {'label': 'Histogram', 'value': 'histogram'},
                             {'label': 'Correlation Graphic', 'value': 'correlation_graphic'}
                         ],
                         value='boxplot'),
            dcc.Graph(id='eda-graph'),

            # "Feature Selection"
            html.H4('Feature Selection'),
            html.P('Choose features:'),
            dcc.Checklist(id='feature-selection-list', 
                          options=[{'label': col, 'value': col} for col in df_new.columns[2:]], 
                          value=[]),
            html.Button('Apply Selection', id='apply-selection', n_clicks=0),
            html.Div(id='feature-methods', children=[
                html.Button('KBest', id='kbest-button', n_clicks=0, style={'display': 'none'}),  # Escondido até ativar
                html.Button('Ensemble Method (Random Forest)', id='random-forest-button', n_clicks=0, style={'display': 'none'}),
                dcc.Graph(id='feature-graph')
            ])
        ])
    elif tab == 'tab-2':
        return html.Div([
        html.H4('IST Power Forecast in CivilPav (kWh)'),
        
        # Dropdown to choose forecasting model
        dcc.Dropdown(
            id='forecast-model-dropdown',
            options=[
                {'label': 'Extreme Gradient Boost', 'value': 'XGB'},
                {'label': 'Neural Network', 'value': 'NN'},
                {'label': 'Random Forest', 'value': 'RF'},
                {'label': 'Bootstrapping', 'value': 'BT'}
            ],
            value='XGB' # Initial value
        ),

        # Plot
        dcc.Graph(id='forecast-graph')
    ])
    elif tab == 'tab-3':
        return html.Div([
        html.H4('IST Power Forecast Error Metrics'),

        # Dropdown for selecting the forecasting method
        dcc.Dropdown(
            id='forecast-method-dropdown',
            options=[
                {'label': 'Extreme Gradient Boost', 'value': 'XGB'},
                {'label': 'Neural Network', 'value': 'NN'},
                {'label': 'Random Forest', 'value': 'RF'},
                {'label': 'Bootstrapping', 'value': 'BT'}
            ],
            value='XGB',  # Default method
            clearable=False
        ),

        # Dropdown for selecting the error metrics to display
        dcc.Dropdown(
            id='metrics-dropdown',
            options=[],  # Will be populated dynamically
            value=[],  # Initially empty
            multi=True
        ),

        # Metrics table
        html.Div(id='metrics-table')
    ])

# "IST Raw Data 2019" Plot
@app.callback(
    [Output('variable-graph', 'figure')],
    Input('variable-dropdown', 'value'))
def update_graphs(selected_variable):
    line_fig = px.line(df_2019, x='Date', y=selected_variable, labels={'Date': 'Date', selected_variable: selected_variable})
    return [line_fig]

# "Exploratory Analysis" Plot
@app.callback(
    Output('eda-graph', 'figure'),
    [Input('eda-dropdown', 'value'),
     Input('variable-dropdown', 'value')])
def update_eda_graph(selected_graph, selected_variable):
    if selected_graph == 'boxplot':
        return px.box(df_2019, y=selected_variable, labels={selected_variable: selected_variable})
    elif selected_graph == 'histogram':
        return px.histogram(df_2019, x=selected_variable, nbins=30, labels={selected_variable: selected_variable})
    elif selected_graph == 'correlation_graphic':
        return px.scatter(df_2019, x=selected_variable, y='Civil (kWh)', labels={selected_variable: selected_variable, 'Civil (kWh)': 'Power'})

# "Feature Selection" Plot
# Selecting Features
@app.callback(
    [Output('kbest-button', 'style'),
     Output('random-forest-button', 'style')],
    Input('apply-selection', 'n_clicks'),
    State('feature-selection-list', 'value')
)
def show_feature_methods(n_clicks, selected_features):
    if n_clicks > 0 and selected_features:
        return [{'display': 'inline-block'}, {'display': 'inline-block'}]
    return [{'display': 'none'}, {'display': 'none'}]

# Features importance according to each method
@app.callback(
    Output('feature-graph', 'figure'),
    [Input('kbest-button', 'n_clicks'),
     Input('random-forest-button', 'n_clicks')],
    State('feature-selection-list', 'value'),
    prevent_initial_call=True
)
def apply_feature_selection(n_clicks_kbest, n_clicks_rf, selected_features):
    ctx = dash.callback_context  # Get context to see which button was clicked

    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]  # Get ID of the button clicked

    if not selected_features:
        return px.bar(title="No features selected! Please choose at least one feature.")

    try:
        # Prepare Data
        df_selected = df_new[['Civil (kWh)'] + selected_features].copy()
        df_selected = df_selected.apply(pd.to_numeric, errors='coerce').dropna()
        Y_clean = df_selected['Civil (kWh)'].values
        X_clean = df_selected[selected_features].values

        if button_id == 'kbest-button':
            # Apply SelectKBest
            features = SelectKBest(score_func=f_regression, k=min(len(selected_features), X_clean.shape[1]))
            fit = features.fit(X_clean, Y_clean)
            scores = fit.scores_
            title = "Feature Selection Scores (KBest)"

        elif button_id == 'random-forest-button':
            # Apply Random Forest
            model = RandomForestRegressor()
            model.fit(X_clean, Y_clean)
            scores = model.feature_importances_
            title = "Feature Selection Scores (Random Forest)"

        # Create Graph
        fig = px.bar(
            x=selected_features,
            y=scores,
            labels={'x': 'Features', 'y': 'Score'},
            title=title
        )
        return fig

    except Exception as e:
        return px.bar(title=f"Error: {str(e)}")

# Forecasting
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('forecast-model-dropdown', 'value')
)
def update_forecast_graph(selected_model):
    # Create dataframe according to chosen model
    if selected_model == 'XGB':
        df_forecast = pd.DataFrame({'Date': df_new['Date'].values, 'Forecast': y_pred_2019_XGB})
        title = "Extreme Gradient Boost Forecast"
    elif selected_model == 'NN':
        df_forecast = pd.DataFrame({'Date': df_new['Date'].values, 'Forecast': y_pred_2019_NN})
        title = "Neural Network Forecast"
    elif selected_model == 'RF':
        df_forecast = pd.DataFrame({'Date': df_new['Date'].values, 'Forecast': y_pred_2019_RF})
        title = "Random Forest Forecast"
    elif selected_model == 'BT':
        df_forecast = pd.DataFrame({'Date': df_new['Date'].values, 'Forecast': y_pred_2019_BT})
        title = "Bootstrapping Forecast"

    # Merge
    df_results = pd.merge(df_real, df_forecast, on='Date')

    # Create plot
    fig = px.line(df_results, x='Date', y=['Real values', 'Forecast'],
                  labels={'Date': 'Date', 'Forecast': 'Forecasted Values'},
                  title=title)
    
    return fig

# Error Metrics
@app.callback(
    [Output('metrics-dropdown', 'options'),
     Output('metrics-dropdown', 'value'),
     Output('metrics-table', 'children')],
    Input('forecast-method-dropdown', 'value'),
    Input('metrics-dropdown', 'value')
)
def update_metrics(selected_method, selected_metrics):
    # Dictionary to map methods to their corresponding metrics
    method_metrics = {
        'XGB': [MAE_XGB, MBE_XGB, MSE_XGB, RMSE_XGB, cvRMSE_XGB, NMBE_XGB],
        'NN': [MAE_NN, MBE_NN, MSE_NN, RMSE_NN, cvRMSE_NN, NMBE_NN],
        'RF': [MAE_RF, MBE_RF, MSE_RF, RMSE_RF, cvRMSE_RF, NMBE_RF],
        'BT': [MAE_BT, MBE_BT, MSE_BT, RMSE_BT, cvRMSE_BT, NMBE_BT]
    }

    # Compute the new df_metrics dynamically
    df_metrics = pd.DataFrame({
        "Métrica": ["MAE", "MBE", "MSE", "RMSE", "cvRMSE", "NMBE"],
        "Valor": method_metrics[selected_method]
    })

    # Create options for dropdown
    metric_options = [{'label': metric, 'value': metric} for metric in df_metrics["Métrica"]]

    # Ensure selected metrics are still valid
    selected_metrics = [m for m in selected_metrics if m in df_metrics["Métrica"].values]
    if not selected_metrics:  # If none are selected, set default values
        selected_metrics = ["MAE", "RMSE"]

    # Filter df_metrics based on selection
    df_filtered = df_metrics[df_metrics["Métrica"].isin(selected_metrics)]

    # Create the table
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df_filtered.columns])),
        html.Tbody([
            html.Tr([html.Td(df_filtered.iloc[i][col]) for col in df_filtered.columns])
            for i in range(len(df_filtered))
        ])
    ])

    return metric_options, selected_metrics, table

###
if __name__ == '__main__':
    app.run()
