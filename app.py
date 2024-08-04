import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import argparse
import seaborn as sns
import numpy as np
import colorcet as cc
import gseapy as gp

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run Dash app with a specified CSV file.')
parser.add_argument('file_path', type=str, help='Path to the CSV file')
args = parser.parse_args()

# Load the dataset
data = pd.read_csv(args.file_path)

# Generate a large color palette for TF_motif
tf_motifs = data['TF_motif'].unique()
num_tf_motifs = len(tf_motifs)
tf_motif_cmap = sns.color_palette(cc.glasbey, n_colors=num_tf_motifs)
tf_motif_colors = {tf_motifs[i]: tuple(tf_motif_cmap[i]) + (0.8,) for i in range(num_tf_motifs)}

# Generate a gradient color palette for time values from 1 to 10
time_values = data["time"].unique()
num_time_values = len(time_values)
time_cmap = sns.color_palette("coolwarm", n_colors=num_time_values)
time_colors = {str(time_values[i]): tuple(time_cmap[i]) + (0.8,) for i in range(num_time_values)}

# Fixed colors for direction
direction_colors = {
    'pos': (44, 160, 44, 0.8),  # Green
    'neg': (214, 39, 40, 0.8)  # Red
}

# Combine all color palettes
color_palette = {**tf_motif_colors, **time_colors, **direction_colors}

# Background color (same as the dummy nodes and links)
background_color = 'rgba(255, 255, 255, 0.0)'

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Layout of the app
app.layout = html.Div([
    html.H1("Interactive Sankey Diagram"),
    html.Div([
        html.Div([
            html.Label("Select left TF_motif:"),
            dcc.Dropdown(
                id='left_tf_motif_filter',
                options=[{'label': tf, 'value': tf} for tf in data['TF_motif'].unique()],
                multi=True
            ),
            html.Label("Select Direction:"),
            dcc.Dropdown(
                id='left_direction_filter',
                options=[{'label': dir, 'value': dir} for dir in data['direction'].unique()],
                multi=True
            ),
            html.Label("Select Time:"),
            dcc.Dropdown(
                id='left_time_filter',
                options=[{'label': time, 'value': time} for time in data['time'].unique()],
                multi=True
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Select right TF_motif:"),
            dcc.Dropdown(
                id='right_tf_motif_filter',
                options=[{'label': tf, 'value': tf} for tf in data['TF_motif'].unique()],
                multi=True
            ),
            html.Label("Select Direction:"),
            dcc.Dropdown(
                id='right_direction_filter',
                options=[{'label': dir, 'value': dir} for dir in data['direction'].unique()],
                multi=True
            ),
            html.Label("Select Time:"),
            dcc.Dropdown(
                id='right_time_filter',
                options=[{'label': time, 'value': time} for time in data['time'].unique()],
                multi=True
            ),
            html.Label("Select right and left join mode:"),
            dcc.RadioItems(
                id='join_type',
                options=['inner', 'outer'],
                value='outer',
                inline=True),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'width': '80%', 'margin': '0 auto', 'display': 'flex', 'justifyContent': 'space-between'}),
    html.Div([
        html.Label("Minimum Score Threshold:"),
        dcc.Slider(
            id='score_threshold',
            min=0,
            max=round(data['score'].abs().max(), 1),
            step=0.1,
            value=0,
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '50%', 'margin': '10px auto'}),
    dcc.Graph(
        id='sankey_diagram',
        style={'height': '100vh', 'width': '100vw'}  # Adjust these values as needed
    ),
    html.Div([
        html.Label("Select Gene Set:"),
        dcc.Dropdown(
            id='gene_set_filter',
            options=[
                {'label': 'GO Molecular Function 2015', 'value': 'GO_Molecular_Function_2015'},
                {'label': 'GO Cellular Component 2015', 'value': 'GO_Cellular_Component_2015'},
                {'label': 'GO Biological Process 2015', 'value': 'GO_Biological_Process_2015'}
            ],
            multi=False
        ),
    ], style={'width': '50%', 'margin': '10px auto'}),
    dcc.Graph(
        id='go_enrichment_plot',
        style={'height': '80vh', 'width': '90vw'}  # Adjust these values as needed
    )
], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})



# Helper function to blend TF_motif color with time greyscale
def blend_colors(tf_color, time_color, alpha=0.7):
    tf_rgb = np.array(tf_color[:3] + (0.4,))
    time_rgb = np.array(time_color[:3] + (0.4,))
    blended_rgb = (1 - alpha) * tf_rgb + alpha * time_rgb
    return "rgba" + str(tuple(blended_rgb))


# Callback to update the Sankey diagram based on filters
@app.callback(
    [Output('sankey_diagram', 'figure'),
     Output('go_enrichment_plot', 'figure')],
    [Input('left_tf_motif_filter', 'value'),
     Input('left_direction_filter', 'value'),
     Input('left_time_filter', 'value'),
     Input('score_threshold', 'value'),
     Input('right_tf_motif_filter', 'value'),
     Input('right_direction_filter', 'value'),
     Input('right_time_filter', 'value'),
     Input('join_type', 'value'),
     Input('gene_set_filter', 'value')]
)
def update_sankey(left_tf_motif_filter, left_direction_filter, left_time_filter, score_threshold, right_tf_motif_filter, right_direction_filter, right_time_filter, join_type, gene_set_filter):
    if not left_tf_motif_filter:
        return go.Figure(), go.Figure()  # Return empty figures if no left TF_motif is selected

    left_direction_filter = left_direction_filter if left_direction_filter else ["pos", "neg"]
    left_time_filter = left_time_filter if left_time_filter else [0,1,2,3,4,5,6,7,8,9]

    left_genes = set(data.loc[data['TF_motif'].isin(left_tf_motif_filter)& (data['score'].abs() >= score_threshold) & (data['direction'].isin(left_direction_filter)) & (data['time'].isin(left_time_filter)), "gene"].unique().tolist())

    if right_tf_motif_filter:
        right_direction_filter = right_direction_filter if right_direction_filter else ["pos", "neg"]
        right_time_filter = right_time_filter if right_time_filter else [0,1,2,3,4,5,6,7,8,9]
        right_genes = set(data.loc[data['TF_motif'].isin(right_tf_motif_filter) & (data['score'].abs() >= score_threshold) & (data['direction'].isin(right_direction_filter)) & (data['time'].isin(right_time_filter)), "gene"].unique().tolist())
        gene_join_filter = left_genes.union(right_genes) if join_type == "outer" else left_genes.intersection(right_genes)
    else:
        gene_join_filter = left_genes

    filtered_data = data[data['TF_motif'].isin(left_tf_motif_filter) &
                         data['gene'].isin(gene_join_filter) &
                          (data['score'].abs() >= score_threshold) &
                          (data['direction'].isin(left_direction_filter)) &
                          (data['time'].isin(left_time_filter))]

    df_summary = filtered_data.groupby(['TF_motif', 'direction', 'time', 'gene']).agg({'score': 'sum'}).reset_index()
    df_summary['score'] = df_summary['score'].abs()

    nodes = list(pd.concat([
        df_summary['TF_motif'],
        df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}", axis=1),
        df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}_{x['time']}", axis=1),
        df_summary['gene']
    ]).unique())

    if right_tf_motif_filter:
        right_filtered_data = data[data['TF_motif'].isin(right_tf_motif_filter) &
                                   data['gene'].isin(gene_join_filter) &
                                    (data['score'].abs() >= score_threshold) &
                                      data['direction'].isin(right_direction_filter) &
                                        data['time'].isin(right_time_filter)].assign(TF_motif = lambda x: " " + x["TF_motif"])
        right_df_summary = right_filtered_data.groupby(['TF_motif', 'direction', 'time', 'gene']).agg({'score': 'sum'}).reset_index()
        right_df_summary['score'] = right_df_summary['score'].abs()

        nodes += list(pd.concat([
            right_df_summary['TF_motif'],
            right_df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}", axis=1),
            right_df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}_{x['time']}", axis=1),
            right_df_summary['gene']
        ]).unique())

    nodes = list(pd.Series(nodes).unique())
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
            node_colors.append("rgba" + str(color_palette[node.split("_")[-1]]))
        elif node.strip() in color_palette.keys():
            node_colors.append("rgba" + str(color_palette[node.strip()]))
        else:
            node_colors.append('rgba(0, 0, 0, 0.8)')  # Default color for genes

    for _, row in df_summary.iterrows():
        tf_motif = row['TF_motif']
        direction = f"{tf_motif}_{row['direction']}"
        time = f"{direction}_{row['time']}"
        gene = row['gene']
        score = row['score']

        links['source'].append(node_indices[tf_motif])
        links['target'].append(node_indices[direction])
        links['value'].append(score)
        links['color'].append("rgba" + str(color_palette.get(tf_motif, (0, 0, 0, 0.4))))

        links['source'].append(node_indices[direction])
        links['target'].append(node_indices[time])
        links['value'].append(score)
        links['color'].append("rgba" + str(color_palette.get(tf_motif, (0, 0, 0, 0.4))))

        links['source'].append(node_indices[time])
        links['target'].append(node_indices[gene])
        links['value'].append(score)
        blended_color = blend_colors(tf_motif_colors[tf_motif], time_colors[str(row['time'])])
        links['color'].append(blended_color)

    if right_tf_motif_filter:
        right_gene_set = set(right_df_summary['gene'])
        left_gene_set = set(df_summary['gene'])
        unconnected_left_genes = left_gene_set - right_gene_set
        unconnected_right_genes = right_gene_set - left_gene_set

        # Add dummy nodes and links for unconnected left genes
        for gene in unconnected_left_genes:
            dummy_tf = ''
            dummy_direction = ' '
            dummy_time = '  '

            if dummy_tf not in nodes:
                nodes.append(dummy_tf)
                nodes.append(dummy_direction)
                nodes.append(dummy_time)
                node_indices[dummy_tf] = len(node_indices)
                node_indices[dummy_direction] = len(node_indices)
                node_indices[dummy_time] = len(node_indices)
                node_colors.append(background_color)
                node_colors.append(background_color)
                node_colors.append(background_color)

            links['source'].append(node_indices[gene])
            links['target'].append(node_indices[dummy_time])
            links['value'].append(0.000001)  # Arbitrary value for dummy links
            links['color'].append(background_color)

            links['source'].append(node_indices[dummy_time])
            links['target'].append(node_indices[dummy_direction])
            links['value'].append(0.000001)
            links['color'].append(background_color)

            links['source'].append(node_indices[dummy_direction])
            links['target'].append(node_indices[dummy_tf])
            links['value'].append(0.000001)
            links['color'].append(background_color)

        # Add dummy nodes and links for unconnected right genes
        for gene in unconnected_right_genes:
            dummy_tf = '    '
            dummy_direction = '     '
            dummy_time = '      '

            if dummy_tf not in nodes:
                nodes.append(dummy_tf)
                nodes.append(dummy_direction)
                nodes.append(dummy_time)
                node_indices[dummy_tf] = len(node_indices)
                node_indices[dummy_direction] = len(node_indices)
                node_indices[dummy_time] = len(node_indices)
                node_colors.append(background_color)
                node_colors.append(background_color)
                node_colors.append(background_color)

            links['source'].append(node_indices[dummy_tf])
            links['target'].append(node_indices[dummy_direction])
            links['value'].append(0.000001)
            links['color'].append(background_color)

            links['source'].append(node_indices[dummy_direction])
            links['target'].append(node_indices[dummy_time])
            links['value'].append(0.000001)
            links['color'].append(background_color)

            links['source'].append(node_indices[dummy_time])
            links['target'].append(node_indices[gene])
            links['value'].append(0.000001)
            links['color'].append(background_color)

        for _, row in right_df_summary.iterrows():
            tf_motif = row['TF_motif']
            direction = f"{tf_motif}_{row['direction']}"
            time = f"{direction}_{row['time']}"
            gene = row['gene']
            score = row['score']

            if gene in node_indices:
                links['source'].append(node_indices[gene])
                links['target'].append(node_indices[time])
                links['value'].append(score)
                links['color'].append("rgba" + str(color_palette.get(tf_motif, (0, 0, 0, 0.4))))

                links['source'].append(node_indices[time])
                links['target'].append(node_indices[direction])
                links['value'].append(score)
                links['color'].append("rgba" + str(color_palette.get(tf_motif, (0, 0, 0, 0.4))))

                links['source'].append(node_indices[direction])
                links['target'].append(node_indices[tf_motif])
                links['value'].append(score)
                blended_color = blend_colors(tf_motif_colors[tf_motif.strip()], time_colors[str(row['time'])])
                links['color'].append(blended_color)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            align="center"
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    )])

    # Perform GO enrichment analysis using gseapy
    if gene_set_filter:
        try:
            go_results = gp.enrichr(gene_list=list(gene_join_filter),
                                    gene_sets=gene_set_filter,
                                    background=list(data['gene'].unique()),
                                    organism="human",
                                    outdir=None)
            go_results_df = go_results.res2d
            go_results_df['log10_pvalue'] = -np.log10(go_results_df['P-value'])
            go_results_df['Odds Ratio'] = go_results_df['Odds Ratio'].replace({np.inf: go_results_df['Odds Ratio'].max() })

            top_terms = go_results_df[go_results_df["P-value"] < 0.1].sort_values(by="P-value").head(20)

            go_fig = go.Figure()

            go_fig.add_trace(go.Bar(
                x=top_terms['log10_pvalue'],
                y=top_terms['Term'],
                orientation='h',
                marker=dict(
                    color=top_terms['Odds Ratio'],
                    colorscale='Viridis',
                    colorbar=dict(title='Odds Ratio')
                ),
                hovertemplate=
                '<b>Term:</b> %{y}<br>'+
                '<b>-log10(p-value):</b> %{x}<br>'+
                '<b>Odds Ratio:</b> %{marker.color}<br>'+
                '<b>Genes:</b> %{customdata}<extra></extra>',
                customdata=top_terms['Genes'].str.split(';').tolist()                
            ))

            go_fig.update_layout(
                title='Top 20 GO Terms',
                xaxis_title='-log10(pvalue)',
                yaxis_title='GO Term',
                yaxis=dict(autorange="reversed"),
                template='plotly_white'
            )
        except Exception as e:
            print(f"Error performing enrichment analysis: {e}")
            go_fig = go.Figure()
    else:
        go_fig = go.Figure()

    return fig, go_fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
