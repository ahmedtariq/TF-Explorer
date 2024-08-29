import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import ks_2samp, ttest_ind
import colorcet as cc
import gseapy as gp
import os

# Load the dataset
file_path = os.getenv('FILE_PATH', 'q_dir_motif_gene_shap_lag.csv')
data = pd.read_csv(file_path)

tfcluster = pd.read_csv("https://jaspar.elixir.no/static/clustering/2024/vertebrates/CORE/interactive_trees/clusters.tab", sep='\t')\
.loc[:, ["cluster", "id", "name"]].assign(name = lambda x: x['name'].str.split(","))\
.assign(id = lambda x: x['id'].str.split(",")).explode(['id','name']).reset_index(drop=True)\
.assign(name = lambda x: x['name'].str.split("::",expand=True)[0].str.upper())

# Generate color palettes
def generate_color_palettes(data):
    tf_motifs = data['TF_motif'].unique()
    num_tf_motifs = len(tf_motifs)
    tf_motif_cmap = sns.color_palette(cc.glasbey, n_colors=num_tf_motifs)
    tf_motif_colors = {tf_motifs[i]: tuple(tf_motif_cmap[i]) + (0.8,) for i in range(num_tf_motifs)}

    time_values = data["time"].unique()
    num_time_values = len(time_values)
    time_cmap = sns.color_palette("coolwarm", n_colors=num_time_values)
    time_colors = {str(time_values[i]): tuple(time_cmap[i]) + (0.8,) for i in range(num_time_values)}

    direction_colors = {
        'pos': (44, 160, 44, 0.8),
        'neg': (214, 39, 40, 0.8)
    }

    color_palette = {**tf_motif_colors, **time_colors, **direction_colors}
    return color_palette, tf_motif_colors, time_colors, direction_colors

color_palette, tf_motif_colors, time_colors, direction_colors = generate_color_palettes(data)
background_color = 'rgba(255, 255, 255, 0.0)'

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    dcc.Store(id='stored_arules_df'),  # Store component to hold the buffered arules data
    html.Div([
        html.Div([
            html.A(
                html.Img(src="https://icb.uni-saarland.de/wp-content/uploads/2020/10/icb_banner.png", style={'height': '100px', 'border-radius': '10px'}),
                href="https://icb.uni-saarland.de/"
            )
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'middle', 'padding': '10px'}),
        html.Div([
            html.Div([
                html.A("Ahmed T Osman", href="mailto:ahos00001@stud.uni-saarland.de", style={'color': '#007bff', 'fontSize': '20px', 'fontWeight': 'bold', 'textDecoration': 'none'}),
                html.Br(),
                html.A("LinkedIn Profile", href="https://www.linkedin.com/in/atosman", style={'color': '#007bff', 'fontSize': '16px', 'textDecoration': 'none'}),
            ], style={'textAlign': 'right', 'padding': '10px'})
        ], style={'width': '80%', 'display': 'inline-block', 'verticalAlign': 'middle'})
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '20px 0', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #dee2e6'}),
    html.H1("TF Explorer"),
    dcc.Tabs([
            dcc.Tab(label='TF Co-regulation', children=[
                html.Div([
                    html.Label("Minimum Score Threshold:"),
                    dcc.Slider(
                        id='tabCo_score_threshold',
                        min=0.5,
                        max=round(data['score'].abs().max(), 1),
                        step=0.1,
                        value=1,
                        marks={(i+0.00001)/10: {"label": str(round((i+0.00001)/10, 1))} for i in range(0, 200, 5) if i/10 < data['score'].abs().max()},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Label("Minimum Support Threshold:"),
                    dcc.Slider(
                        id='tabCo_support_threshold',
                        min=0.05,
                        max=1,
                        step=0.01,
                        value=0.06,  # Default value for support threshold
                        marks={i/100: {"label": str(i/100)} for i in range(0, 101, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '50%', 'margin': '10px auto'}),
                html.Div([
                    html.Label("Minimum Adjusted Lift Threshold:"),
                    dcc.Slider(
                        id='tabCo_peak_adj_lift_threshold',
                        min=0,  # These will be dynamically updated
                        max=1,  # These will be dynamically updated
                        step=0.1,
                        value=0,  # Default range selection
                        marks={},  # Dynamic marks
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Label("Select Time:"),
                    dcc.Dropdown(
                        id='tabCo_time_filter',
                        options=[{'label': str(i), 'value': i} for i in range(0, 10)],
                        multi=True
                    ),
                    html.Label("Select Direction:"),
                    dcc.Dropdown(
                        id='tabCo_direction_filter',
                        options=[{'label': 'pos', 'value': 'pos'}, {'label': 'neg', 'value': 'neg'}],
                        multi=True
                    ),
                ], style={'width': '50%', 'margin': '10px auto', 'borderTop': '2px solid #dee2e6', 'paddingTop': '20px'}),
                dcc.Graph(
                    id='tf_co_regulation_graph',
                    style={'height': '100vh', 'width': '100vw'}  # Adjust as needed
                )
        ]),
        dcc.Tab(label='Transcription Factor', children=[
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
                        options=[],
                        multi=True,
                        value=['pos', 'neg']
                    ),
                    html.Label("Select Time:"),
                    dcc.Dropdown(
                        id='left_time_filter',
                        options=[],
                        multi=True,
                        disabled=True
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
                        options=[],
                        multi=True,
                        value=['pos', 'neg']
                    ),
                    html.Label("Select Time:"),
                    dcc.Dropdown(
                        id='right_time_filter',
                        options=[],
                        multi=True,
                        disabled=True
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
                    step = 0.1,
                    value = 0,
                    marks = {(i+0.00001)/10: {"label" : str(round((i+0.00001)/10,1))}for i in range(0,200,5) if i/10 < data['score'].abs().max()},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '50%', 'margin': '10px auto'}),
            dcc.Graph(
                id='sankey_diagram',
                style={'height': '100vh', 'width': '100vw'}  # Adjust these values as needed
            ),
            html.Div([
                html.Label("Background:"),
                dcc.RadioItems(
                    id='background_choice',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Filtered by TF', 'value': 'filtered'}
                    ],
                    value='all',
                    inline=True
                )
            ], style={'width': '80%', 'margin': '20px auto', 'display': 'flex', 'justifyContent': 'center'}),
            html.Div([
                dcc.Graph(
                    id='distance_density_plot',
                    style={'height': '80vh', 'width': '40vw', 'display': 'inline-block'}  # Adjust these values as needed
                ),
                html.Div([
                    html.Label("Select Gene Set:"),
                    dcc.Dropdown(
                        id='gene_set_filter',
                        style={'width': '20vw'},
                        options=[
                            {'label': 'GO Molecular Function 2015', 'value': 'GO_Molecular_Function_2015'},
                            {'label': 'GO Cellular Component 2015', 'value': 'GO_Cellular_Component_2015'},
                            {'label': 'GO Biological Process 2015', 'value': 'GO_Biological_Process_2015'}
                        ],
                        multi=False
                    ),
                    dcc.Graph(
                        id='go_enrichment_plot',
                        style={'height': '70vh', 'width': '50vw'}  # Adjust these values as needed
                    )
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'space-around'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'justifyContent': 'space-around'}),
            dcc.Graph(
                id='dist_lag_violin_plot',
                style={'height': '60vh', 'width': '100vw'}  # Adjust the height as needed
            )
        ]),
        dcc.Tab(label='Gene', children=[
            html.Div([
                html.Label("Select Gene:"),
                dcc.Dropdown(
                    id='tabG_gene_filter',
                    options=[{'label': gene, 'value': gene} for gene in data['gene'].unique()],
                    multi=True,
                    value=[]
                ),
                html.Label("Select Direction:"),
                dcc.Dropdown(
                    id='tabG_direction_filter',
                    options=[],
                    multi=True,
                    value=['pos', 'neg']
                ),
                html.Label("Select Time:"),
                dcc.Dropdown(
                    id='tabG_time_filter',
                    options=[],
                    multi=True,
                    disabled=True
                ),
                html.Label("Minimum Score Threshold:"),
                dcc.Slider(
                    id='tabG_score_threshold',
                    min=0,
                    max=round(data['score'].abs().max(), 1),
                    step=0.1,
                    value=0,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '50%', 'margin': '10px auto'}),
            dcc.Graph(
                id='tabG_sankey_diagram',
                style={'height': '100vh', 'width': '100vw'}  # Adjust these values as needed
            ),
            html.Div([
                html.Label("Background:"),
                dcc.RadioItems(
                    id='tabG_background_choice',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Filtered by TF', 'value': 'filtered'}
                    ],
                    value='all',
                    inline=True
                )
            ], style={'width': '80%', 'margin': '20px auto', 'display': 'flex', 'justifyContent': 'center'}),
            dcc.Graph(
                id='tabG_distance_density_plot',
                style={'height': '50vh', 'width': '100vw', 'display': 'inline-block'}  # Adjust these values as needed
            )
        ])
    ])
], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})


# Helper function to blend TF_motif color with time greyscale
def blend_colors(tf_color, time_color, alpha=0.7):
    tf_rgb = np.array(tf_color[:3] + (0.4,))
    time_rgb = np.array(time_color[:3] + (0.4,))
    blended_rgb = (1 - alpha) * tf_rgb + alpha * time_rgb
    return "rgba" + str(tuple(blended_rgb))

# Callbacks to update filters based on selected TF_motifs and directions
@app.callback(
    [Output('left_direction_filter', 'options'),
     Output('left_direction_filter', 'value'),
     Output('left_direction_filter', 'disabled')],
    [Input('left_tf_motif_filter', 'value')]
)
def update_left_direction_filter(selected_tf_motifs):
    if not selected_tf_motifs:
        return [], ['pos', 'neg'], True

    filtered_data = data[data['TF_motif'].isin(selected_tf_motifs)]
    directions = [{'label': dir, 'value': dir} for dir in filtered_data['direction'].unique()]

    return directions, ['pos', 'neg'], False

@app.callback(
    [Output('right_direction_filter', 'options'),
     Output('right_direction_filter', 'value'),
     Output('right_direction_filter', 'disabled')],
    [Input('right_tf_motif_filter', 'value')]
)
def update_right_direction_filter(selected_tf_motifs):
    if not selected_tf_motifs:
        return [], ['pos', 'neg'], True

    filtered_data = data[data['TF_motif'].isin(selected_tf_motifs)]
    directions = [{'label': dir, 'value': dir} for dir in filtered_data['direction'].unique()]

    return directions, ['pos', 'neg'], False

@app.callback(
    [Output('left_time_filter', 'options'),
     Output('left_time_filter', 'disabled')],
    [Input('left_tf_motif_filter', 'value'),
     Input('left_direction_filter', 'value')]
)
def update_left_time_filter(selected_tf_motifs, selected_directions):
    if not selected_tf_motifs or not selected_directions:
        return [], True

    filtered_data = data[(data['TF_motif'].isin(selected_tf_motifs)) & (data['direction'].isin(selected_directions))]
    times = [{'label': time, 'value': time} for time in filtered_data['time'].unique()]

    return times, False

@app.callback(
    [Output('right_time_filter', 'options'),
     Output('right_time_filter', 'disabled')],
    [Input('right_tf_motif_filter', 'value'),
     Input('right_direction_filter', 'value')]
)
def update_right_time_filter(selected_tf_motifs, selected_directions):
    if not selected_tf_motifs or not selected_directions:
        return [], True

    filtered_data = data[(data['TF_motif'].isin(selected_tf_motifs)) & (data['direction'].isin(selected_directions))]
    times = [{'label': time, 'value': time} for time in filtered_data['time'].unique()]

    return times, False

# Callbacks to update Gene filters based on selected TF_motifs and directions

@app.callback(
    [Output('tabG_direction_filter', 'options'),
     Output('tabG_direction_filter', 'value'),
     Output('tabG_direction_filter', 'disabled')],
    [Input('tabG_gene_filter', 'value')]
)
def update_tabG_direction_filter(selected_genes):
    if not selected_genes:
        return [], ['pos', 'neg'], True

    filtered_data = data[data['gene'].isin(selected_genes)]
    directions = [{'label': dir, 'value': dir} for dir in filtered_data['direction'].unique()]

    return directions, ['pos', 'neg'], False

@app.callback(
    [Output('tabG_time_filter', 'options'),
     Output('tabG_time_filter', 'disabled')],
    [Input('tabG_gene_filter', 'value'),
     Input('tabG_direction_filter', 'value')]
)
def update_tabG_time_filter(selected_genes, selected_directions):
    if not selected_genes or not selected_directions:
        return [], True

    filtered_data = data[(data['gene'].isin(selected_genes)) & (data['direction'].isin(selected_directions))]
    times = [{'label': time, 'value': time} for time in filtered_data['time'].unique()]

    return times, False

# Callback to update the Gene filter in the Gene tab based on clicks on the Sankey diagram
@app.callback(
    Output('tabG_gene_filter', 'value'),
    [Input('sankey_diagram', 'clickData')],
    [State('tabG_gene_filter', 'value')]
)
def update_gene_filter_from_sankey(clickData, current_genes):
    if clickData is None:
        return current_genes

    clicked_node = clickData['points'][0]['label']
    if ' ' in clicked_node:  # This condition may need to be adapted based on your node labels
        return current_genes  # Not a gene node, return current filter without changes

    # If clicked_node is a gene, update the gene filter
    if current_genes is None:
        current_genes = []

    if clicked_node not in current_genes:
        current_genes.append(clicked_node)

    return current_genes

@app.callback(
    Output('stored_arules_df', 'data'),
    [Input('tabCo_score_threshold', 'value'),
     Input('tabCo_support_threshold', 'value')]
)
def update_arules_data(tabCo_score_threshold, tabCo_support_threshold):
    # Generate the association rules dataframe
    allq_arules_df = make_arules(data, tabCo_score_threshold, tabCo_support_threshold)
    
    # Store the dataframe in a dictionary format to store in dcc.Store
    return allq_arules_df.to_dict('records')

@app.callback(
    [
        Output('tabCo_peak_adj_lift_threshold', 'min'),
        Output('tabCo_peak_adj_lift_threshold', 'max'),
        Output('tabCo_peak_adj_lift_threshold', 'value'),
        Output('tabCo_peak_adj_lift_threshold', 'marks')
    ],
    Input('stored_arules_df', 'data')
)
def update_peak_adj_lift_slider(stored_arules_df):
    if not stored_arules_df:
        return 0, 1, [0, 1], {}
    
    df = pd.DataFrame(stored_arules_df)
    
    min_lift = df['peak_adj_lift'].min()
    max_lift = df['peak_adj_lift'].max()
    
    # Generate marks for the slider
    marks = {val: {"label": str(val)} for val in range(0, int(max_lift+1), 1)}
    
    return min_lift, max_lift, min_lift, marks

@app.callback(
    Output('tf_co_regulation_graph', 'figure'),
    [
        Input('stored_arules_df', 'data'),  # Use the stored arules data as input
        Input('tabCo_peak_adj_lift_threshold', 'value'),
        Input('tabCo_time_filter', 'value'),
        Input('tabCo_direction_filter', 'value'),
        Input('tf_co_regulation_graph', 'clickData')
    ]
)
def update_tf_co_regulation_graph(stored_arules_df, tabCo_peak_adj_lift_threshold, tabCo_time_filter, tabCo_direction_filter, clickData):
    tabCo_direction_filter = tabCo_direction_filter if tabCo_direction_filter else ["pos", "neg"]
    tabCo_time_filter = tabCo_time_filter if tabCo_time_filter else [0,1,2,3,4,5,6,7,8,9]
    # Convert the stored data back to a DataFrame
    allq_arules_df = pd.DataFrame(stored_arules_df)
    
    # Generate the graph using the existing logic
    fig = generate_tf_co_regulation_graph(data, tfcluster, allq_arules_df, tabCo_peak_adj_lift_threshold, tabCo_time_filter, tabCo_direction_filter)

    if clickData is not None:
        # Extract the clicked node's label
        clicked_node_label = clickData['points'][0]['text']
        clicked_node_hovertext = clickData['points'][0]['hovertext']

        # Highlight the clicked node and its connected edges and nodes
        fig = highlight_node_and_edges(fig, clicked_node_hovertext, allq_arules_df)

    return fig

# Main callback to update the Sankey diagram, distance density plot, and GO enrichment plot
@app.callback(
    [Output('sankey_diagram', 'figure'),
     Output('distance_density_plot', 'figure'),
     Output('go_enrichment_plot', 'figure'),
     Output('dist_lag_violin_plot', 'figure')],
    [Input('left_tf_motif_filter', 'value'),
     Input('left_direction_filter', 'value'),
     Input('left_time_filter', 'value'),
     Input('score_threshold', 'value'),
     Input('right_tf_motif_filter', 'value'),
     Input('right_direction_filter', 'value'),
     Input('right_time_filter', 'value'),
     Input('join_type', 'value'),
     Input('gene_set_filter', 'value'),
     Input('background_choice', 'value')]
)
def update_graphs(left_tf_motif_filter, left_direction_filter, left_time_filter, score_threshold, right_tf_motif_filter, right_direction_filter, right_time_filter, join_type, gene_set_filter, background_choice):
    if not left_tf_motif_filter:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()  # Return empty figures if no left TF_motif is selected

    left_direction_filter = left_direction_filter if left_direction_filter else ["pos", "neg"]
    left_time_filter = left_time_filter if left_time_filter else [0,1,2,3,4,5,6,7,8,9]

    left_genes = set(data.loc[data['TF_motif'].isin(left_tf_motif_filter) & 
                             (data['score'].abs() >= score_threshold) & 
                             (data['direction'].isin(left_direction_filter)) & 
                             (data['time'].isin(left_time_filter)), 
                             "gene"].unique().tolist())

    if right_tf_motif_filter:
        right_direction_filter = right_direction_filter if right_direction_filter else ["pos", "neg"]
        right_time_filter = right_time_filter if right_time_filter else [0,1,2,3,4,5,6,7,8,9]
        right_genes = set(data.loc[data['TF_motif'].isin(right_tf_motif_filter) & 
                                  (data['score'].abs() >= score_threshold) & 
                                  (data['direction'].isin(right_direction_filter)) & 
                                  (data['time'].isin(right_time_filter)), 
                                  "gene"].unique().tolist())
        gene_join_filter = left_genes.union(right_genes) if join_type == "outer" else left_genes.intersection(right_genes)
    else:
        gene_join_filter = left_genes

    filtered_data = data[data['TF_motif'].isin(left_tf_motif_filter) &
                         data['gene'].isin(gene_join_filter) &
                          (data['score'].abs() >= score_threshold) &
                          (data['direction'].isin(left_direction_filter)) &
                          (data['time'].isin(left_time_filter))]
    
    df_summary = filtered_data.groupby(['TF_motif', 'direction', 'time', 'peak' ,'gene']).agg({'score': 'sum'}).reset_index()
    df_summary['score'] = df_summary['score'].abs()
    if right_tf_motif_filter:
        right_filtered_data = data[data['TF_motif'].isin(right_tf_motif_filter) &
                                   data['gene'].isin(gene_join_filter) &
                                    (data['score'].abs() >= score_threshold) &
                                      data['direction'].isin(right_direction_filter) &
                                        data['time'].isin(right_time_filter)].assign(TF_motif = lambda x: " " + x["TF_motif"])
        right_df_summary = right_filtered_data.groupby(['TF_motif', 'direction', 'time', 'peak', 'gene']).agg({'score': 'sum'}).reset_index()
        right_df_summary['score'] = right_df_summary['score'].abs()
    else :
        right_filtered_data = pd.DataFrame(columns=filtered_data.columns)
        right_df_summary = None

    nodes, node_indices, links, node_colors = generate_sankey_nodes_and_links(df_summary, right_df_summary, right_tf_motif_filter, tf_motif_colors, time_colors, background_color, color_palette)

    sankey_fig = create_sankey_figure(nodes, links, node_colors)
    distance_density_fig = create_distance_density_plot(filtered_data, right_filtered_data, background_choice, left_tf_motif_filter, right_tf_motif_filter, data)
    go_enrichment_fig = create_go_enrichment_plot(gene_set_filter, gene_join_filter, background_choice, left_tf_motif_filter, right_tf_motif_filter, data)
    dist_lag_fig = create_dist_lag_violin_plot(filtered_data, right_filtered_data, background_choice, left_tf_motif_filter, right_tf_motif_filter, data)

    return sankey_fig, distance_density_fig, go_enrichment_fig, dist_lag_fig

# Main callback to update the Sankey diagram and distance density plot for Gene tab
@app.callback(
    [Output('tabG_sankey_diagram', 'figure'),
     Output('tabG_distance_density_plot', 'figure')],
    [Input('tabG_gene_filter', 'value'),
     Input('tabG_direction_filter', 'value'),
     Input('tabG_time_filter', 'value'),
     Input('tabG_score_threshold', 'value'),
     Input('tabG_background_choice', 'value')]
)
def update_gene_tab_graphs(tabG_gene_filter, tabG_direction_filter, tabG_time_filter, score_threshold, background_choice):
    if not tabG_gene_filter:
        return go.Figure(), go.Figure()  # Return empty figures if no gene is selected

    tabG_direction_filter = tabG_direction_filter if tabG_direction_filter else ["pos", "neg"]
    tabG_time_filter = tabG_time_filter if tabG_time_filter else [0,1,2,3,4,5,6,7,8,9]

    filtered_data = data[data['gene'].isin(tabG_gene_filter) &
                          (data['score'].abs() >= score_threshold) &
                          (data['direction'].isin(tabG_direction_filter)) &
                          (data['time'].isin(tabG_time_filter))]

    df_summary = filtered_data.groupby(['TF_motif', 'direction', 'time', 'peak' ,'gene']).agg({'score': 'sum'}).reset_index()
    df_summary['score'] = df_summary['score'].abs()

    filterd_TF_motf = df_summary['TF_motif'].unique().tolist()
    left_filterd_TF_motf = filterd_TF_motf[:int(len(filterd_TF_motf)/2)]
    right_filterd_TF_motf = filterd_TF_motf[int(len(filterd_TF_motf)/2):]

    df_summary_left = df_summary[df_summary["TF_motif"].isin(left_filterd_TF_motf)]
    df_summary_right = df_summary[df_summary["TF_motif"].isin(right_filterd_TF_motf)]

    nodes, node_indices, links, node_colors = generate_sankey_nodes_and_links(df_summary_left, df_summary_right, True, tf_motif_colors, time_colors, background_color, color_palette)

    sankey_fig = create_sankey_figure(nodes, links, node_colors)
    distance_density_fig = create_distance_density_plot(filtered_data, background_choice, left_filterd_TF_motf, right_filterd_TF_motf, data)

    return sankey_fig, distance_density_fig


def make_arules(data, tabCo_score_threshold, tabCo_support_threshold):
    data["TF_motif"] = data["TF_motif"].str.split('::',expand=True)[0].str.split('(',expand=True)[0].str.upper()
    # Apply the score threshold filter
    data = data[data["score"].abs() > tabCo_score_threshold]

    # Start with your provided graph generation code
    perGene_itemSet_df = data.assign(motif_direction_time = lambda x: x["TF_motif"] + "_"+x["direction"] + "_" + x["time"].astype(str)).\
    groupby("gene")["motif_direction_time"].agg(list)

    te = TransactionEncoder()
    te_ary = te.fit(perGene_itemSet_df).transform(perGene_itemSet_df)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=tabCo_support_threshold, use_colnames=True)

    perGene_ass_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

    # Filter rules with 1 antecedents & 1 consequents
    perGene_ass_rules_df["antecedents_len"] = perGene_ass_rules_df.apply(lambda x: len(x["antecedents"]),axis=1)
    perGene_ass_rules_df["consequents_len"] = perGene_ass_rules_df.apply(lambda x: len(x["consequents"]),axis=1)
    perGene_ass_rules_df = perGene_ass_rules_df.query("antecedents_len == 1 & consequents_len == 1").sort_values(by="lift", ascending= False)

    # Transforming rules to string
    perGene_ass_rules_df['antecedents'] = perGene_ass_rules_df['antecedents'].apply(lambda x: ''.join(x))
    perGene_ass_rules_df['consequents'] = perGene_ass_rules_df['consequents'].apply(lambda x: ''.join(x))

    # Function to identify if a row should be removed
    def should_remove(row, df):
        # Find potential matching rows
        matches = df[(df['antecedents'].astype(str) == row['consequents']) & (df['consequents'] == row['antecedents'])]
        
        # If matches are found and confidence of the current row is lower than the matched row
        if not matches.empty:
            if row['confidence'] < matches['confidence'].max():
                return True
        return False

    # Removing duplicate rules because of direction
    perGene_ass_rules_df = perGene_ass_rules_df.loc[~perGene_ass_rules_df.apply(should_remove, axis=1, args=(perGene_ass_rules_df,)),:]

    # Extracting info from rules
    perGene_ass_rules_df = perGene_ass_rules_df.assign(antecedents_TF_motif = lambda x: x["antecedents"].str.split("_",expand=True)[0] )\
    .assign(antecedents_dir = lambda x: x["antecedents"].str.split("_",expand=True)[1])\
    .assign(antecedents_time = lambda x: x["antecedents"].str.split("_",expand=True)[2])\
    .assign(consequents_TF_motif = lambda x: x["consequents"].str.split("_",expand=True)[0])\
    .assign(consequents_dir = lambda x: x["consequents"].str.split("_",expand=True)[1])\
    .assign(consequents_time = lambda x: x["consequents"].str.split("_",expand=True)[2])

    def calc_jaccard_apply(x):
        peak_ant = data.loc[(data["TF_motif"] == x["antecedents_TF_motif"])  & (data["time"] == int(x["antecedents_time"])),"peak"].drop_duplicates()
        peak_con = data.loc[(data["TF_motif"] == x["consequents_TF_motif"]) & (data["time"] == int(x["consequents_time"])) ,"peak"].drop_duplicates()
        intersection = peak_ant.isin(peak_con).sum()
        union = pd.concat([peak_ant,peak_con],axis=0).nunique()
        return intersection/union

    perGene_ass_rules_df["TF_peak_jaccard"] = perGene_ass_rules_df.apply(calc_jaccard_apply,axis=1)
    perGene_ass_rules_df = perGene_ass_rules_df.assign(peak_adj_lift = lambda x: (1 - x["TF_peak_jaccard"]) * x["lift"])
    
    return perGene_ass_rules_df


def generate_tf_co_regulation_graph(data, tfcluster, allq_arules_df, tabCo_peak_adj_lift_threshold, tabCo_time_filter, tabCo_direction_filter):

    data["TF_motif"] = data["TF_motif"].str.split('::',expand=True)[0].str.split('(',expand=True)[0].str.upper()

    allq_arules_df = allq_arules_df[
        (allq_arules_df['peak_adj_lift'] >= tabCo_peak_adj_lift_threshold) &
        (allq_arules_df['antecedents_time'].astype(int).isin(tabCo_time_filter)) &
        (allq_arules_df['consequents_time'].astype(int).isin(tabCo_time_filter)) &
        (allq_arules_df['antecedents_dir'].isin(tabCo_direction_filter)) &
        (allq_arules_df['consequents_dir'].isin(tabCo_direction_filter))
    ]

    # Generate the igraph network visualization 
    # Note: We will create the plot and return it
    def fade_to_white(x,start_r = 0, start_g = 150 , start_b = 0 , alpha = 0.6):
        # Ensure x is between 0 and 1
        x = max(0, min(1, x))
        x = 1 - x
        
        
        # Ending color (rgba(255, 255, 255, 1))
        end_r = 255
        end_g = 255
        end_b = 255
        
        # Interpolate each channel
        r = int(start_r + (end_r - start_r) * x)
        g = int(start_g + (end_g - start_g) * x)
        b = int(start_b + (end_b - start_b) * x)
        a = alpha
        
        return f'rgba({r},{g},{b},{a})'

    # arules data provided by mlxtend
    df = allq_arules_df.query("confidence < 1.1").loc[:,["antecedents","antecedents_TF_motif", "antecedents_dir", "antecedents_time", "consequents","consequents_TF_motif",
                                "consequents_dir", "consequents_time", "antecedent support", "consequent support", "support",
                                "confidence", "lift", "TF_peak_jaccard", "peak_adj_lift" ]]

    # node cluster info
    node_cluster_df = pd.merge(pd.concat([df["antecedents_TF_motif"],df["consequents_TF_motif"]],axis=0).drop_duplicates().reset_index(drop=True).rename("TF_motif").to_frame(),
            tfcluster[["cluster", "name"]], left_on="TF_motif", right_on="name", how="left").drop("name",axis=1)
    node_cluster_df['cluster'] = node_cluster_df['cluster'].fillna('no_cluster')

    # Generate a list of colors for cluster
    unique_clusters = node_cluster_df["cluster"].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_clusters)).colors  # You can choose other colormaps
    cluster_color_map = {cluster: f'rgba({color[0]*255},{color[1]*255},{color[2]*255},0.9)' for cluster, color in zip(unique_clusters, colors)}
    cluster_color_map["no_cluster"] = 'rgba(0,0,0,0.9)'


    # Create a directed graph
    graph = ig.Graph(directed=True)

    # Add nodes (TF motifs)
    nodes = pd.concat([df["antecedents"], df["consequents"]]).unique()
    graph.add_vertices(nodes)
    graph.vs["TF_motif"] = [node["name"].split("_")[0] for node in graph.vs]
    graph.vs["dir"] = [node["name"].split("_")[1] for node in graph.vs]
    graph.vs["time"] = [node["name"].split("_")[2] for node in graph.vs]
    graph.vs["cluster"] = [node_cluster_df.loc[node_cluster_df["TF_motif"] == node["TF_motif"],"cluster"].iloc[0]  for node in graph.vs]
    graph.vs["color"] = [cluster_color_map[node["cluster"]] for node in graph.vs]


    # Add edges
    for index, row in df.iterrows():
        color = 'green' if row['antecedents_dir'] == row['consequents_dir'] else 'red'
        graph.add_edge(row['antecedents'], row['consequents'], 
                    color=color, support=row['support'], confidence=row['confidence'], lift=row['lift'],
                    antecedent = row["antecedents_TF_motif"], consequent = row["consequents_TF_motif"],
                    same_direction = row["antecedents_dir"] == row["consequents_dir"], TF_peak_jaccard = row["TF_peak_jaccard"], peak_adj_lift = row["peak_adj_lift"])

    # Get edge attributes
    g_edge_tooltips = ["{}-{}<br>Support: {:.4f}<br>Confidence: {:.4f}<br>Lift: {:.4f}<br>peak_jaccard: {:.4f}".format(edge['antecedent'],edge['consequent'],edge['support'], edge['confidence'], edge['lift'], edge['TF_peak_jaccard']) for edge in graph.es]
    g_color = ["green" if edge["same_direction"] else "red"  for edge in graph.es]
    g_width = [round(edge["peak_adj_lift"],1) for edge in graph.es]


    # Layout using Kamada-Kawai algorithm
    layout = graph.layout('kk')


    # Create the Plotly figure
    g_edge_x = []
    g_edge_y = []

    for e in graph.es:
        x0, y0 = layout[e.source]
        x1, y1 = layout[e.target]
        g_edge_x.extend([x0, x1, None])
        g_edge_y.extend([y0, y1, None])

    g_edge_trace_ls = []
    for w,c in set(zip(g_width,g_color)):
        index_filter = [i for i,val in enumerate(zip(g_width,g_color)) if (val[0] == w) & (val[1] == c)]
        g_edge_trace_ls.append(go.Scatter(
            x=[ix for i in index_filter for ix in g_edge_x[i*3:i*3+3]], y=[ix for i in index_filter for ix in g_edge_y[i*3:i*3+3]],
            line=dict(width=2, color=fade_to_white( max(w/ np.max(g_width),0.1),start_r= 0 if c == "green" else 150,start_g= 150 if c == "green" else 0, alpha=0.8)),
            hoverinfo='text',
            text=[g_edge_tooltips[i] for i in index_filter],
            mode='lines')
            )

    g_edge_middle_trace = go.Scatter(
        x=[(g_edge_x[i] + g_edge_x[i+1])/2 for i in range(0,len(g_edge_x),3)], y=[(g_edge_y[i] + g_edge_y[i+1])/2 for i in range(0,len(g_edge_y),3)],
        mode='markers',
        hoverinfo='text',
        text=g_edge_tooltips,
        hoverlabel=dict(
            bgcolor="rgba(230,236,246,0.4)"  # Background color when hovering
        ),
        marker=go.Marker(
            opacity=0
        ))


    node_x = [layout[v.index][0] for v in graph.vs]
    node_y = [layout[v.index][1] for v in graph.vs]
    node_text = ["{}<br>{}<br>time {}<br>{}".format(node["TF_motif"],node["dir"],node["time"],node["cluster"]) for node in graph.vs]
    node_TF_motif = graph.vs["TF_motif"]
    node_cluster = graph.vs["cluster"]
    node_time = graph.vs["time"]
    node_color = graph.vs["color"]
    node_shape = graph.vs["time"]
    node_outline_color = ["green" if node["dir"] == "pos" else "red" for node in graph.vs]



    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_TF_motif,
        hoverinfo='text',
        hovertext=node_text,
        textposition='top center',
        textfont=dict(
            size=10,  # Adjust the font size as needed
            weight = "normal"
        ),
        marker=dict(
            size=8,
            color=node_color,
            symbol=node_shape,
            line=dict(color=node_outline_color, width=0.75)
            ),
        hoverlabel=dict(
                bgcolor=node_color  # Background color when hovering
            )
        )

    fig = go.Figure(data=g_edge_trace_ls + [g_edge_middle_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        plot_bgcolor='white',
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=40),
                        title="Association Rules Network",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )


    # Returning the figure
    return fig

def highlight_node_and_edges(fig, clicked_node_hovertext, allq_arules_df):
    # converting hover text to _ speprated string
    def hover_to_str(hovertext):
        return "_".join(hovertext.replace("time ","").split("<br>")[:-1])
    
    clicked_node_dir_time = hover_to_str(clicked_node_hovertext)
    # Identify the nodes and edges connected to the clicked node
    connected_edges = allq_arules_df[(allq_arules_df['antecedents'] == clicked_node_dir_time ) | 
                                     (allq_arules_df['consequents'] == clicked_node_dir_time )]
    
    
    connected_nodes = set(connected_edges['antecedents']).union(set(connected_edges['consequents']))

    # Update the graph's node and edge traces to highlight the relevant parts
    for trace in fig['data']:
        if trace['mode'] == 'markers+text':
            # Highlight nodes
            trace.update(
                marker=dict(
                    size=[
                        16 if hover_to_str(hovertext) == clicked_node_dir_time else 12 if (hover_to_str(hovertext) in connected_nodes) else 8
                        for hovertext in trace['hovertext']
                    ],
                    line=dict(
                        color=[
                            'darkred' if ((outline_color == "red") | (outline_color == 'darkred')) else "darkgreen" if (hover_to_str(hovertext) in connected_nodes) else outline_color
                            for outline_color, hovertext in zip( trace['marker']['line']['color'], trace['hovertext'])
                        ],
                        width=[
                            4 if hover_to_str(hovertext) == clicked_node_dir_time else 2 if (hover_to_str(hovertext) in connected_nodes) else 0.75
                            for hovertext in trace['hovertext']
                        ]
                    )
                ),
                textfont=dict(
                    weight = [
                        "bold" if hover_to_str(hovertext) in connected_nodes else "normal"
                        for hovertext in trace['hovertext']

                    ],
                    color = [
                        "darkorange" if hover_to_str(hovertext) == clicked_node_dir_time else "darkblue" if (hover_to_str(hovertext) in connected_nodes) else "rgb(0,0,0)"
                        for hovertext in trace['hovertext']

                    ]

                )


            )

    return fig






def generate_sankey_nodes_and_links(df_summary, right_df_summary, right_tf_motif_filter, tf_motif_colors, time_colors, background_color, color_palette):
    nodes = list(pd.concat([
        df_summary['TF_motif'],
        df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}", axis=1),
        df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}_{x['time']}", axis=1),
        df_summary['gene']
    ]).unique())

    if right_tf_motif_filter:
        nodes += list(pd.concat([
            right_df_summary['TF_motif'],
            right_df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}", axis=1),
            right_df_summary.apply(lambda x: f"{x['TF_motif']}_{x['direction']}_{x['time']}", axis=1),
            right_df_summary['gene']
        ]).unique())

    nodes = list(pd.Series(nodes).unique())
    node_indices = {node: i for i, node in enumerate(nodes)}

    links = create_sankey_links(df_summary, node_indices, tf_motif_colors, time_colors, color_palette, None, left = True)
    node_colors = create_sankey_node_colors(nodes, direction_colors, time_colors, color_palette)

    if right_tf_motif_filter:
        right_gene_set = set(right_df_summary['gene'])
        left_gene_set = set(df_summary['gene'])
        unconnected_left_genes = left_gene_set - right_gene_set
        unconnected_right_genes = right_gene_set - left_gene_set

        add_dummy_nodes_and_links(unconnected_left_genes, unconnected_right_genes, nodes, node_indices, links, background_color, node_colors)
        links = create_sankey_links(right_df_summary, node_indices, tf_motif_colors, time_colors, color_palette, links, left= False)
    
    return nodes, node_indices, links, node_colors



def create_sankey_links(df_summary, node_indices, tf_motif_colors, time_colors, color_palette, links, left = True):
    if links is None:
        links = {
            'source': [],
            'target': [],
            'value': [],
            'color': [],
            'label': [],
            'hovercolor': []
        }
    if left:
        a = 'source'
        b = 'target'
    else:
        a = 'target'
        b = 'source'        

    for _, row in df_summary.iterrows():
        tf_motif = row['TF_motif']
        direction = f"{tf_motif}_{row['direction']}"
        time = f"{direction}_{row['time']}"
        gene = row['gene']
        peak = row['peak']
        score = row['score']

        links[a].append(node_indices[tf_motif])
        links[b].append(node_indices[direction])
        links['value'].append(score)
        links['color'].append("rgba" + str(color_palette.get(tf_motif.strip(), (0, 0, 0, 0.4))))
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

        links[a].append(node_indices[direction])
        links[b].append(node_indices[time])
        links['value'].append(score)
        links['color'].append("rgba" + str(color_palette.get(tf_motif.strip(), (0, 0, 0, 0.4))))
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

        links[a].append(node_indices[time])
        links[b].append(node_indices[gene])
        links['value'].append(score)
        blended_color = blend_colors(tf_motif_colors[tf_motif.strip()], time_colors[str(row['time'])])
        links['color'].append(blended_color)
        links['label'].append('peak: ' + peak)
        links['hovercolor'].append("mediumturquoise")

    return links

def create_sankey_node_colors(nodes, direction_colors, time_colors, color_palette):
    node_colors = []
    for node in nodes:
        if (node.split("_")[-1] in direction_colors.keys()) | (node.split("_")[-1] in time_colors.keys()):
            node_colors.append("rgba" + str(color_palette[node.split("_")[-1]]))
        elif node.strip() in color_palette.keys():
            node_colors.append("rgba" + str(color_palette[node.strip()]))
        else:
            node_colors.append('rgba(0, 0, 0, 0.8)')  # Default color for genes
    return node_colors

def add_dummy_nodes_and_links(unconnected_left_genes, unconnected_right_genes, nodes, node_indices, links, background_color, node_colors):
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
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

        links['source'].append(node_indices[dummy_time])
        links['target'].append(node_indices[dummy_direction])
        links['value'].append(0.000001)
        links['color'].append(background_color)
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

        links['source'].append(node_indices[dummy_direction])
        links['target'].append(node_indices[dummy_tf])
        links['value'].append(0.000001)
        links['color'].append(background_color)
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

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
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

        links['source'].append(node_indices[dummy_direction])
        links['target'].append(node_indices[dummy_time])
        links['value'].append(0.000001)
        links['color'].append(background_color)
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

        links['source'].append(node_indices[dummy_time])
        links['target'].append(node_indices[gene])
        links['value'].append(0.000001)
        links['color'].append(background_color)
        links['label'].append('')
        links['hovercolor'].append(links['color'][-1])

def create_sankey_figure(nodes, links, node_colors):
    return go.Figure(data=[go.Sankey(
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
            color=links['color'],
            label=links['label'],
            hovercolor=links['hovercolor']
        ),
        arrangement='fixed'
    )])

def create_distance_density_plot(filtered_data, right_filtered_data, background_choice, left_tf_motif_filter, right_tf_motif_filter, data):
    if not filtered_data.empty:
        try:
            background_distances = data.drop_duplicates("peak")['distance'] if background_choice == "all" else data[data["TF_motif"].isin(left_tf_motif_filter + (right_tf_motif_filter or [""]))].drop_duplicates("peak")['distance']
            filtered_distances = filtered_data.drop_duplicates("peak")['distance']
            right_filtered_distances = right_filtered_data.drop_duplicates("peak")['distance']
            rug_text = (filtered_data.drop_duplicates("peak")['gene'] + "  " + filtered_data.drop_duplicates("peak")['peak']).tolist()
            right_rug_text = (right_filtered_data.drop_duplicates("peak")['gene'] + "  " + right_filtered_data.drop_duplicates("peak")['peak']).tolist()

            _, l_p_value = ks_2samp(filtered_distances, background_distances)
            left_p_value_text = "<b>*</b>" if l_p_value < 0.05 else "<b>ns</b>"

            if not right_filtered_data.empty:
                _, r_p_value = ks_2samp(right_filtered_distances, background_distances)
                right_p_value_text = "<b>*</b>" if r_p_value < 0.05 else "<b>ns</b>"

                fig_density = ff.create_distplot(
                    [filtered_distances, right_filtered_distances, background_distances], ['Left Filtered Peaks ' + left_p_value_text, 'Right Filtered Peaks ' + right_p_value_text, 'Background'],
                    bin_size=10000, rug_text=[rug_text, right_rug_text, None], colors=['rgba(0, 0, 255, 0.8)', 'rgb(0, 255, 0, 0.8)', 'rgba(200, 200, 200, 0.6)'])
            else:
                fig_density = ff.create_distplot(
                    [filtered_distances, background_distances], ['Left Filtered Peaks ' + left_p_value_text, 'Background'],
                    bin_size=10000, rug_text=[rug_text, None], colors=['rgba(0, 0, 255,0.8)',  'rgba(200, 200, 200, 0.6)'])
            fig_density.update_xaxes(range=[-200000, 200000])
            fig_density.add_vline(x=0, line_width=2, line_dash="dash", line_color="black", annotation_text="TSS", annotation_position="top right")
            fig_density.update_layout(
                height=800,
                margin=dict(t=50, b=50, l=50, r=50),
                yaxis2=dict(
                    domain=[0, 0.1]
                ),
                yaxis=dict(
                    domain=[0.15, 1]
                )
            )
        except Exception as e:
            print(f"Error performing distance analysis: {e}")
            fig_density = go.Figure()
    else:
        fig_density = go.Figure()
    return fig_density

def create_go_enrichment_plot(gene_set_filter, gene_join_filter, background_choice, left_tf_motif_filter, right_tf_motif_filter, data):
    if gene_set_filter:
        try:
            background_genes = data.drop_duplicates("gene")['gene'] if background_choice == "all" else data[data["TF_motif"].isin(left_tf_motif_filter + (right_tf_motif_filter or [""]))].drop_duplicates("gene")['gene']
            go_results = gp.enrichr(gene_list=list(gene_join_filter),
                                    gene_sets=gene_set_filter,
                                    background=list(background_genes),
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
                template='plotly_white',
                height=750
            )
        except Exception as e:
            print(f"Error performing enrichment analysis: {e}")
            go_fig = go.Figure()
    else:
        go_fig = go.Figure()
    return go_fig

def create_dist_lag_violin_plot(filtered_data, right_filtered_data, background_choice, left_tf_motif_filter, right_tf_motif_filter, data):
    if not filtered_data.empty:
        try:

            # Assuming data and filtered_data are pandas DataFrames

            fig_dist_lag = go.Figure()

            data = data.sort_values(by="time")  if background_choice == "all" else data[data["TF_motif"].isin(left_tf_motif_filter + (right_tf_motif_filter or [""]))].sort_values(by="time")
            filtered_data = pd.concat([filtered_data, right_filtered_data],axis=0).sort_values(by="time") if not right_filtered_data.empty else filtered_data.sort_values(by="time")

            fig_dist_lag.add_trace(go.Violin(x=filtered_data['time'].astype(str),
                                    y=filtered_data['mean_lag'],
                                    legendgroup='Filtered', scalegroup='Filtered', name='Filtered',
                                    side='negative',
                                    pointpos=-0.5, # where to position points
                                    line_color='lightseagreen')
                    )
            fig_dist_lag.add_trace(go.Violin(x=data['time'][data["time"].isin(pd.unique(filtered_data['time']))].astype(str),
                                    y=data['mean_lag'][data["time"].isin(pd.unique(filtered_data['time']))],
                                    legendgroup='Background', scalegroup='Background', name='Background',
                                    side='positive',
                                    pointpos=0.5,
                                    line_color='mediumpurple')
                    )

            # Perform t-tests and add asterisks for significant differences
            max_y_value = max(filtered_data['mean_lag'].max(), data['mean_lag'].max()) + 15

            for x_index, time_point in enumerate(pd.unique(filtered_data['time'])):
                filtered_lags = filtered_data['mean_lag'][filtered_data['time'] == time_point]
                background_lags = data['mean_lag'][data['time'] == time_point]
                
                # Perform t-test
                t_stat, p_value = ttest_ind(filtered_lags, background_lags, equal_var=False)
                
                # If significant, add the asterisk
                # time_point = time_point if time_point != pd.unique(filtered_data['time'])[-1] else time_point -1
                if (p_value < 0.05) & (p_value >= 0.01):
                    fig_dist_lag.add_annotation(x=x_index, y=max_y_value,
                                    text="*", showarrow=False,
                                    font=dict(size=20, color="black")) 
                elif p_value < 0.01:
                    fig_dist_lag.add_annotation(x=x_index, y=max_y_value,
                                    text="**", showarrow=False,
                                    font=dict(size=20, color="black")) 
                elif p_value >= 0.05 :
                    fig_dist_lag.add_annotation(x=x_index, y=max_y_value,
                                    text="ns", showarrow=False,
                                    font=dict(size=20, color="black"))        

            # Update layout for visibility
            fig_dist_lag.update_traces(meanline_visible=True, box_visible=True,
                            points='all', jitter=0.05)  # add some jitter on points for better visibility
            fig_dist_lag.update_layout(
                title_text="Mean Delay Distribution",
                violingap=0, violingroupgap=0, violinmode='overlay'
            )

            return fig_dist_lag

        except Exception as e:
            print(f"Error performing mean lag analysis: {e}")
            fig_dist_lag = go.Figure()
    else:
        fig_dist_lag = go.Figure()
    return fig_dist_lag

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
