import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.Div(), width=1),  # Top-left empty corner
        dbc.Col(html.Div(), width=1),  # Top-right empty corner
        dbc.Col(html.Div(), width=1),  # Top-right empty corner
    ]),

    dbc.Row([
        dbc.Col(html.Div(), width=1),  # Left empty column
        dbc.Col(html.Div([
            html.Label('Label 1'),
            dcc.Input(type='text', placeholder='Field 1')
        ]), width=4),
        dbc.Col(html.Div([
            html.Label('Label 2'),
            dcc.Input(type='text', placeholder='Field 2')
        ]), width=4),
        dbc.Col(html.Div(), width=1),  # Right empty column
    ]),

    dbc.Row([
        dbc.Col(html.Div(), width=1),  # Left empty column
        dbc.Col(html.Div([
            html.Label('Label 3'),
            dcc.Input(type='text', placeholder='Field 3')
        ]), width=4),
        dbc.Col(html.Div([
            html.Label('Label 4'),
            dcc.Input(type='text', placeholder='Field 4')
        ]), width=4),
        dbc.Col(html.Div(), width=1),  # Right empty column
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(), width=1),  # Bottom-left empty corner
        dbc.Col(html.Div(), width=1),  # Bottom-right empty corner
        dbc.Col(html.Div(), width=1),  # Bottom-right empty corner
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
