import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
import os

# Load the dataset
file_path = os.getenv('FILE_PATH', 'q_dir_motif_gene_shap_lag.csv')
data = pd.read_csv(file_path)
data["TF_motif"] = data["TF_motif"].str.split('::', expand=True)[0].str.split('(', expand=True)[0].str.upper()

tfcluster = pd.read_csv("https://jaspar.elixir.no/static/clustering/2024/vertebrates/CORE/interactive_trees/clusters.tab", sep='\t')\
    .loc[:, ["cluster", "id", "name"]].assign(name=lambda x: x['name'].str.split(","))\
    .assign(id=lambda x: x['id'].str.split(",")).explode(['id', 'name']).reset_index(drop=True)\
    .assign(name=lambda x: x['name'].str.split("::", expand=True)[0].str.upper())

# Join data with cluster information
filter_data = data.assign(score=data["score"].abs()).merge(tfcluster.loc[:, ["cluster", "name"]], how='left', left_on='TF_motif', right_on='name').drop("name", axis=1)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server


# Helper function to create pivot table with hierarchical clustering
def create_pivot_table(filter_data, index_cols, column_cols, value_col, agg_func, cluster_rows, cluster_cols):
    if value_col not in filter_data.columns:
        print(f"[DEBUG] Value column '{value_col}' not in data.")
        return pd.DataFrame()  # Return empty if value_col doesn't exist

    agg_func_dict = {
        "sum": "sum",
        "mean": "mean",
        "std": "std",
        "median": "median",
        "count": "count",
        "nunique": pd.Series.nunique,
    }

    if agg_func not in agg_func_dict:
        print(f"[DEBUG] Aggregation function '{agg_func}' is not supported.")
        return pd.DataFrame()

    try:
        pivot = pd.pivot_table(
            filter_data,
            values=value_col,
            index=index_cols,
            columns=column_cols,
            aggfunc=agg_func_dict[agg_func],
            fill_value=0
        )
    except Exception as e:
        print(f"[DEBUG] Error creating pivot table: {e}")
        return pd.DataFrame()

    # Apply hierarchical clustering if requested
    try:
        if cluster_rows and not pivot.empty:
            row_linkage = linkage(pivot.values, method='ward')
            row_order = leaves_list(row_linkage)
            pivot = pivot.iloc[row_order]

        if cluster_cols and not pivot.empty:
            col_linkage = linkage(pivot.T.values, method='ward')
            col_order = leaves_list(col_linkage)
            pivot = pivot.iloc[:, col_order]
    except Exception as e:
        print(f"[DEBUG] Error applying clustering: {e}")
        return pd.DataFrame()

    return pivot.reset_index().round(2)


# App layout
app.layout = html.Div([
    html.H2("Transcription Factor-Gene Interaction Explorer", style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs", value='tabData', children=[
        dcc.Tab(label='TF-Gene Data and Analysis', value='tabData', children=[
            html.Div([
                html.H3("Search and Explore Interactions"),
                html.Div([
                    html.Details([
                        html.Summary("About the Dataset"),
                        html.P("Explore predicted TF-Gene interactions with metadata such as interaction scores, TF clusters, and distances."),
                        dash_table.DataTable(
                            id='data_description',
                            columns=[
                                {"name": "Column Name", "id": "Column"},
                                {"name": "Description", "id": "Description"}
                            ],
                            data=[
                                {"Column": "TF_motif", "Description": "Transcription Factor (TF) involved in the interaction."},
                                {"Column": "gene", "Description": "Target gene regulated by the TF."},
                                {"Column": "peak", "Description": "Genomic region linking TF and gene."},
                                {"Column": "time", "Description": "Developmental timepoint of interaction."},
                                {"Column": "score", "Description": "Interaction score predicted by the algorithm."},
                                {"Column": "direction", "Description": "Positive or negative regulation of the target gene."},
                                {"Column": "distance", "Description": "Distance from the binding peak to the gene's TSS."},
                                {"Column": "cluster", "Description": "Cluster grouping transcription factors."},
                                {"Column": "mean_lag", "Description": "Average lagged effect between the TF and target gene."},
                            ],
                            style_table={'margin': '10px', 'width': '100%'},
                            style_cell={'textAlign': 'left', 'whiteSpace': 'normal'},
                            style_header={'fontWeight': 'bold'}
                        )
                    ])
                ]),
                dcc.Checklist(
                    options=[{'label': 'Enable Custom Query', 'value': 'advanced'}],
                    id='advanced_filter_toggle',
                    style={'marginBottom': '10px'}
                ),
                html.Div(
                    [
                        dcc.Input(
                            id='filter-query-input',
                            placeholder="Enter query: ({distance} > 5000 or {distance} < -5000) and score > 1",
                            style={'width': '100%', 'display': 'none'}
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Syntax Help",
                                    style={
                                        'color': 'blue',
                                        'textDecoration': 'underline',
                                        'cursor': 'pointer',
                                        'position': 'relative'
                                    },
                                    id='syntax-help'
                                ),
                                html.Div(
                                    [
                                        "Syntax for Queries:",
                                        html.Ul([
                                            html.Li("Surround column names with `{}`."),
                                            html.Li("For numeric data, use `=`, `>`, `<`, `>=`, `<=`."),
                                            html.Li("For text data, use `contains` or `=`."),
                                            html.Li("Combine conditions using `and`, `or`."),
                                        ])
                                    ],
                                    id='tooltip',
                                    style={
                                        'position': 'absolute',
                                        'top': '20px',
                                        'left': '0',
                                        'backgroundColor': '#f9f9f9',
                                        'border': '1px solid #ccc',
                                        'padding': '10px',
                                        'display': 'none',
                                        'zIndex': 10,
                                        'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)',
                                        'fontSize': '12px'
                                    }
                                )
                            ],
                            style={'display': 'none'},
                            id='tooltip-container'
                        )
                    ],
                    id='advanced-filter-container'
                ),
                dash_table.DataTable(
                    id='data_table',
                    columns=[
                        {"name": col, "id": col,
                         "type": "numeric" if np.issubdtype(filter_data[col].dtype, np.number) else "text"}
                        for col in filter_data.columns
                    ],
                    data=filter_data.to_dict('records'),
                    page_action='native',
                    filter_action='native',
                    sort_action='native',
                    page_size=10,
                    style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'scroll'},
                    style_cell={'textAlign': 'center', 'minWidth': '70px', 'width': '70px', 'maxWidth': '150px'}
                ),
                html.Button("Export TF Links Table", id="download_data_table_button", style={'marginTop': '10px'}),
                dcc.Download(id="download_data_table"),
                html.Hr(style={'margin': '20px 0'}),
                html.H3("Pivot Table Analysis"),
                html.Label("Choose Rows for Analysis:"),
                dcc.Dropdown(
                    id='pivot_index',
                    options=[{'label': col, 'value': col} for col in filter_data.columns],
                    multi=True
                ),
                html.Label("Choose Columns for Analysis:"),
                dcc.Dropdown(
                    id='pivot_columns',
                    options=[{'label': col, 'value': col} for col in filter_data.columns],
                    multi=True
                ),
                html.Label("Choose Value to Analyze:"),
                dcc.Dropdown(
                    id='pivot_values',
                    options=[{'label': col, 'value': col} for col in filter_data.columns]
                ),
                html.Label("Aggregation Method:"),
                dcc.Dropdown(
                    id='pivot_aggfunc',
                    options=[]
                ),
                html.Label("Group Rows/Columns:"),
                dcc.Checklist(
                    id='apply_clustering',
                    options=[
                        {'label': 'Cluster Rows', 'value': 'rows'},
                        {'label': 'Cluster Columns', 'value': 'columns'}
                    ]
                ),
                html.Button('Generate Pivot Table', id='generate_pivot', n_clicks=0),
                dash_table.DataTable(
                    id='pivot_table',
                    style_table={'overflowX': 'scroll', 'maxHeight': '500px', 'overflowY': 'scroll'},
                    style_header={'position': 'sticky', 'top': 0, 'backgroundColor': 'white', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'center', 'fontSize': '10px', 'minWidth': '50px', 'maxWidth': '150px'}
                ),
                html.Button("Export Analysis Results", id="download_button", style={'display': 'none'}),
                dcc.Download(id="download_pivot_table")
            ])
        ])
    ])
])



@app.callback(
    [
        Output('filter-query-input', 'style'),
        Output('tooltip-container', 'style')
    ],
    Input('advanced_filter_toggle', 'value')
)
def toggle_advanced_filter(selected):
    if selected and 'advanced' in selected:
        # Advanced filter selected
        return {'width': '100%', 'display': 'block'}, {'display': 'block'}
    else:
        # Default/native filter
        return {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('data_table', 'filter_query'),
    Input('filter-query-input', 'value'),
    State('advanced_filter_toggle', 'value')
)
def apply_advanced_filter(query, selected):
    # Debugging logs to ensure correct inputs
    print(f"[DEBUG] Received query: {query}")
    print(f"[DEBUG] Advanced filter toggle state: {selected}")

    if selected and 'advanced' in selected:
        if query:
            # Return the filter query to the DataTable
            print(f"[DEBUG] Applying filter query: {query}")
            return query
        else:
            print("[DEBUG] No query provided, returning empty string.")
            return ''  # Return an empty query if no input is given
    else:
        print("[DEBUG] Advanced filter not enabled, returning empty string.")
        return ''  # Disable advanced filter if not selected

@app.callback(
    Output("download_data_table", "data"),
    Input("download_data_table_button", "n_clicks"),
    State("data_table", "derived_virtual_data"),
    prevent_initial_call=True
)
def download_data_table(n_clicks, derived_virtual_data):
    filtered_df = pd.DataFrame(derived_virtual_data)
    return dcc.send_data_frame(filtered_df.to_csv, "data_table.csv")

@app.callback(
    Output('tooltip', 'style'),
    Input('syntax-help', 'n_clicks'),
    prevent_initial_call=True
)
def show_tooltip(n_clicks):
    # Toggle the display of the tooltip
    if n_clicks % 2 == 1:
        return {'position': 'absolute', 'top': '20px', 'left': '0',
                'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc',
                'padding': '10px', 'display': 'block', 'zIndex': 10,
                'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)', 'fontSize': '12px'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('pivot_aggfunc', 'options'),
    Input('pivot_values', 'value')
)
def update_aggfunc_options(selected_value_col):
    """
    Dynamically updates the aggregation function options based on the data type
    of the selected "Values" column.
    """
    if not selected_value_col:
        return []
    
    # Check the data type of the selected column
    if selected_value_col in filter_data.select_dtypes(include=np.number).columns:
        # Numeric column
        return [
            {'label': 'Mean', 'value': 'mean'},
            {'label': 'Median', 'value': 'median'},
            {'label': 'Sum', 'value': 'sum'},
            {'label': 'Standard Deviation', 'value': 'std'},
        ]
    else:
        # Non-numeric column
        return [
            {'label': 'Count', 'value': 'count'},
            {'label': 'Unique Count', 'value': 'nunique'},
        ]


@app.callback(
    [
        Output('pivot_table', 'data'),
        Output('pivot_table', 'columns'),
        Output('pivot_table', 'style_data_conditional'),
        Output('download_button', 'style')
    ],
    Input('generate_pivot', 'n_clicks'),
    State('data_table', 'derived_virtual_data'),
    State('pivot_index', 'value'),
    State('pivot_columns', 'value'),
    State('pivot_values', 'value'),
    State('pivot_aggfunc', 'value'),
    State('apply_clustering', 'value')
)
def generate_pivot_table(n_clicks, derived_virtual_data, index_cols, column_cols, value_col, agg_func, apply_clustering):
    print(f"[DEBUG] Generating pivot table... n_clicks={n_clicks}")
    print(f"[DEBUG] Index: {index_cols}, Columns: {column_cols}, Value: {value_col}, Agg Func: {agg_func}, Clustering: {apply_clustering}")

    if not derived_virtual_data:
        print("[DEBUG] Using full dataset as no filtering applied.")
        derived_virtual_data = filter_data.to_dict('records')

    if not index_cols or not value_col or not agg_func:
        print(f"[DEBUG] Missing required inputs. Index: {index_cols}, Value: {value_col}, Agg Func: {agg_func}")
        return [], [], [], {'display': 'none'}

    try:
        filtered_df = pd.DataFrame(derived_virtual_data)
        cluster_rows = 'rows' in apply_clustering
        cluster_cols = 'columns' in apply_clustering

        pivot = create_pivot_table(filtered_df, index_cols, column_cols, value_col, agg_func, cluster_rows, cluster_cols)

        if pivot.empty:
            print("[DEBUG] Pivot table is empty.")
            return [], [], []

        # Prepare columns for the Dash DataTable
        columns = [{"name": str(col), "id": str(col)} for col in pivot.columns]

        # Calculate global min/max for numeric coloring
        numeric_data = pivot.select_dtypes(include=np.number)
        if not numeric_data.empty:
            min_val = numeric_data.min().min()
            max_val = numeric_data.max().max()
        else:
            min_val, max_val = 0, 1

        # Ensure we have a valid range for scaling
        range_val = max_val - min_val if max_val != min_val else 1

        # Generate style_data_conditional for cell-level coloring
        style_data_conditional = []
        for col in pivot.columns:
            if col in numeric_data:
                for row_idx, value in pivot[col].items():
                    scaled_value = (value - min_val) / range_val
                    background_color = f'rgb(255, {255 - int(255 * scaled_value)}, 200)'
                    style_data_conditional.append({
                        'if': {'row_index': row_idx, 'column_id': str(col)},
                        'backgroundColor': background_color,
                        'color': 'black'
                    })

        print("[DEBUG] Pivot table generated successfully.")
        print(pivot)  # Log the pivot table for debugging
        return pivot.to_dict('records'), columns, style_data_conditional, {'display': 'block'}
    except Exception as e:
        print(f"[DEBUG] Error in callback: {e}")
        return [{"Error": str(e)}], [], [], {'display': 'none'}




@app.callback(
    Output("download_pivot_table", "data"),
    Input("download_button", "n_clicks"),
    State('pivot_table', 'data'),
    State('pivot_table', 'columns'),
    prevent_initial_call=True
)
def download_pivot_table(n_clicks, pivot_data, pivot_columns):
    if pivot_data and pivot_columns:
        pivot_df = pd.DataFrame(pivot_data)
        pivot_df.columns = [col["name"] for col in pivot_columns]
        return dcc.send_data_frame(pivot_df.to_csv, "pivot_table.csv")
    return None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
