import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import colorcet as cc

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run Dash app with a specified CSV file.')
parser.add_argument('file_path', type=str, help='Path to the CSV file')
args = parser.parse_args()

# Load the dataset
data = pd.read_csv(args.file_path)

# Generate a large color palette for TF_motif
tf_motifs = data['TF_motif'].unique()
num_tf_motifs = len(tf_motifs)
tf_motif_cmap =  sns.color_palette(cc.glasbey, n_colors=num_tf_motifs)
tf_motif_colors = {tf_motifs[i]: tuple(tf_motif_cmap[i])+ (0.8,) for i in range(num_tf_motifs)}



# Generate a gradient color palette for time values from 1 to 10
time_values = data["time"].unique()
num_time_values = len(time_values)
time_cmap = sns.color_palette(n_colors=num_time_values)
time_colors = {str(time_values[i]): tuple(time_cmap[i])+ (0.8,) for i in range(num_time_values)}

# Fixed colors for direction
direction_colors = {
    'pos': (44, 160, 44, 0.8),  # Green
    'neg': (214, 39, 40,0.8)  # Red
}

# Combine all color palettes
color_palette = {**tf_motif_colors, **time_colors, **direction_colors}

# Create the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Interactive Sankey Diagram"),
    html.Div([
        html.Label("Select TF_motif:"),
        dcc.Dropdown(
            id='tf_motif_filter',
            options=[{'label': tf, 'value': tf} for tf in data['TF_motif'].unique()],
            multi=True
        ),
    ]),
    html.Div([
        html.Label("Select Direction:"),
        dcc.Dropdown(
            id='direction_filter',
            options=[{'label': dir, 'value': dir} for dir in data['direction'].unique()],
            multi=True,
            disabled=True
        ),
    ]),
    html.Div([
        html.Label("Select Time:"),
        dcc.Dropdown(
            id='time_filter',
            options=[{'label': time, 'value': time} for time in data['time'].unique()],
            multi=True,
            disabled=True
        ),
    ]),
    html.Div([
        html.Label("Minimum Score Threshold:"),
        dcc.Slider(
            id='score_threshold',
            min=0,
            max=round(data['score'].abs().max(),1),
            step=0.1,
            value=0
        ),
    ]),
    dcc.Graph(id='sankey_diagram')
])


# Helper function to blend TF_motif color with time greyscale
def blend_colors(tf_color, time_color, alpha=0.7):
    tf_rgb = np.array(tf_color[:3] + (0.4,))
    time_rgb = np.array(time_color[:3] + (0.4,))
    blended_rgb = (1 - alpha) * tf_rgb + alpha * time_rgb
    return "rgba"+ str(tuple(blended_rgb))


# Callback to update the Sankey diagram based on filters
@app.callback(
    Output('sankey_diagram', 'figure'),
    [Input('tf_motif_filter', 'value'),
     Input('score_threshold', 'value')]
)
def update_sankey(tf_motif_filter, score_threshold):
    if not tf_motif_filter:
        return go.Figure()  # Return an empty figure if no TF_motif is selected

    filtered_data = data[data['TF_motif'].isin(tf_motif_filter) & (data['score'].abs() >= score_threshold)]

    df_summary = filtered_data.groupby(['TF_motif', 'direction', 'time', 'gene']).agg({'score': 'sum'}).reset_index()
    df_summary['score'] = df_summary['score'].abs()
    
    nodes = list(pd.concat([
        df_summary['TF_motif'],
        df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}", axis=1),
        df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}_{x['time']}", axis=1),
        df_summary['gene']
    ]).unique())


    node_indices = {node: i for i, node in enumerate(nodes)}

    links = {
        'source': [],
        'target': [],
        'value': [],
        'color': []
    }

    node_colors = []
    for node in nodes:
        if (node.split("_")[-1] in direction_colors.keys()) | (node.split("_")[-1] in time_colors.keys()):
            node_colors.append("rgba"+str(color_palette[node.split("_")[-1]]))
        elif node in color_palette.keys():
            node_colors.append("rgba"+str(color_palette[node]))
        else:
            node_colors.append('rgba(0, 0, 0, 0.8)' )  # Default color for genes

    for _, row in df_summary.iterrows():
        tf_motif = row['TF_motif']
        direction = f"{tf_motif}_{row['direction']}"
        time = f"{direction}_{row['time']}"
        gene = row['gene']
        score = row['score']
        
        links['source'].append(node_indices[tf_motif])
        links['target'].append(node_indices[direction])
        links['value'].append(score)
        links['color'].append("rgba"+str(color_palette.get(tf_motif, (0, 0, 0,0.4) )))
        
        links['source'].append(node_indices[direction])
        links['target'].append(node_indices[time])
        links['value'].append(score)
        links['color'].append("rgba"+str(color_palette.get(tf_motif, (0, 0, 0,0.4) )))
        
        links['source'].append(node_indices[time])
        links['target'].append(node_indices[gene])
        links['value'].append(score)
        blended_color = blend_colors(tf_motif_colors[tf_motif], time_colors[str(row['time'])])
        links['color'].append(blended_color)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    )])

    fig.update_layout(title_text="Interactive Sankey Diagram", font_size=10, height=1200)

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
