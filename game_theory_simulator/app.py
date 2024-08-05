import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def leader_cost(p, q, a, b):
    return a * p ** 2 - b * np.log(1 + p / q)

def follower_cost(q, p, c, d):
    return c * q ** 2 - d * np.log(1 + q / p)

def follower_response(p, c, d, initial_guess=0.1):
    result = minimize(lambda q: follower_cost(q, p, c, d), initial_guess, bounds=[(0.01, None)])
    return result.x[0]

def find_stackelberg_equilibrium(a, b, c, d):
    def objective(p):
        q = follower_response(p, c, d)
        return leader_cost(p, q, a, b)

    initial_guess = 0.1
    bounds = [(0.01, None)]
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_p = result.x[0]
    optimal_q = follower_response(optimal_p, c, d)
    return optimal_p, optimal_q

# Custom styles
custom_styles = {
    'container': {
        'backgroundColor': '#800000',  # Maroon
        'padding': '20px',
        'borderRadius': '10px',
        'color': '#ffffff'  # White text for contrast
    },
    'header': {
        'textAlign': 'center',
        'marginBottom': '20px',
        'color': '#ffa500'  # Orange
    },
    'label': {
        'color': '#ffa500'  # Orange
    },
    'slider_container': {
        'marginBottom': '20px'
    },
    'image': {
        'width': '150px',
        'display': 'block',
        'margin': '0 auto',
        'float': 'right'
    }
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Img(src='/assets/image.png', style=custom_styles['image']),
            html.H1("Game-Theoretic Power Allocation Simulator", style=custom_styles['header'])
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label('Parameter a:', style=custom_styles['label']),
                dcc.Slider(id='a-slider', min=0.1, max=5, step=0.1, value=1, marks={i: str(i) for i in range(1, 6)})
            ], style=custom_styles['slider_container'])
        ], width=6),
        dbc.Col([
            html.Div([
                html.Label('Parameter b:', style=custom_styles['label']),
                dcc.Slider(id='b-slider', min=0.1, max=5, step=0.1, value=2, marks={i: str(i) for i in range(1, 6)})
            ], style=custom_styles['slider_container'])
        ], width=6)
    ], className="my-3"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label('Parameter c:', style=custom_styles['label']),
                dcc.Slider(id='c-slider', min=0.1, max=5, step=0.1, value=1, marks={i: str(i) for i in range(1, 6)})
            ], style=custom_styles['slider_container'])
        ], width=6),
        dbc.Col([
            html.Div([
                html.Label('Parameter d:', style=custom_styles['label']),
                dcc.Slider(id='d-slider', min=0.1, max=5, step=0.1, value=2, marks={i: str(i) for i in range(1, 6)})
            ], style=custom_styles['slider_container'])
        ], width=6)
    ], className="my-3"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='graph')
        ])
    ])
], fluid=True, style=custom_styles['container'])

@app.callback(
    Output('graph', 'figure'),
    [Input('a-slider', 'value'), Input('b-slider', 'value'),
     Input('c-slider', 'value'), Input('d-slider', 'value')]
)
def update_figure(a, b, c, d):
    p_star, q_star = find_stackelberg_equilibrium(a, b, c, d)
    powers = np.linspace(0.01, 3, 100)
    responses = [follower_response(p, c, d) for p in powers]

    return {
        'data': [
            go.Scatter(x=powers, y=responses, mode='lines', name='Follower response'),
            go.Scatter(x=[p_star], y=[q_star], mode='markers', name='Equilibrium Point', marker=dict(color='red'))
        ],
        'layout': go.Layout(
            title='Follower Response vs Leader Power Level',
            xaxis={'title': 'Leader Power Level (p)'},
            yaxis={'title': 'Follower Power Level (q)'},
            template='plotly_dark'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
