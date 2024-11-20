from dash import Dash, dash_table, dcc, html, callback, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc



SEGMENT_METHOD_DROPDOWN_OPTIONS = [
    "Otsu (Global) Binarization",
    "Local Threshold",
    "Pixel Classifier",
    "Ensemble",
    "MaskRCNN",
    "Segment Anything"
]

EDGE_DETECTION_DROPDOWN_OPTIONS = [
    "Local Thresholding",
    "Bright-Dark",
    "None",
]

def get_vertical_label_field(label_text,field_name,field_default="") -> html.Div:
    '''
    Helper function to get vertical stack of label and field for elements
    '''
    return html.Div(children=[
        dbc.Row(
            
                html.Label(label_text),
        ),
        dbc.Row(
            dcc.Input(
                id=field_name,
                type='text',
                placeholder=field_default
            )
        ),
        ],
        className="vertical_label_field")

def get_vertical_dropdown(label_text,dropdown_name,dropdown_options,
) -> html.Div:
    '''
    Get a vertically stacked label and dropdown option
    '''
    return html.Div(children=[
        dbc.Row(
            
                html.Label(label_text),
        ),
        dbc.Row(
            dcc.Dropdown(
                dropdown_options,
                dropdown_options[0],
                id=dropdown_name
            )
        ),
        ],
        className="vertical_label_dropdown"
        )

def get_image_load_div() -> html.Div:
    '''
    Call to return an html div that sets up the image load in divs
    This should all be one function call since they are organized pretty distinctly here!
    '''

    ## Image Window w/ Label
    input_div = html.Div([
        html.Div("Input",id="input_image_label"),
        html.Div(html.Div(id="input_image_div"))
    ]
    )

    ## Left and Right buttons w/ Progress Bar
    navigation_progress = dbc.Row(children=[
        dbc.Col(html.Button("<",id="input_left_arrow",)),
        dbc.Col(dbc.Progress(id="progress_bar")),
        dbc.Col(html.Button(">",id="input_right_arrow",))
    ])

    ## Load Button
    load_button = html.Div(
        [
            dcc.Upload(
                id="load_button",
                children=html.Button('Load Image File(s)'),
                multiple=True
            ),
        ]
    )

    ## Input Fields
    # Fields
    top_bound = get_vertical_label_field("Top Bound","top_bound_field",0)
    bottom_bound = get_vertical_label_field("Bottom Bound","bottom_bound_field",860)
    left_bound = get_vertical_label_field("Left Bound","left_bound_field",0)
    right_bound = get_vertical_label_field("Right Bound","right_bound_field",2560)
    px_to_um = get_vertical_label_field("px to um","px_to_um_field","9.37")

    # for element in [top_bound,bottom_bound,left_bound,right_bound,px_to_um]:
    #     element.className="grid-item"

    # Sorting
    grid_vals = html.Div(
        children=[
            dbc.Row(
                children=[
                    dbc.Col(html.Div(),width=4,),
                    dbc.Col(children=top_bound,width=4),
                    dbc.Col(html.Div(),width=4,)
                ],
                
            ),
            dbc.Row(
                children=[
                    dbc.Col(children=left_bound,width=4),
                    dbc.Col(children=px_to_um,width=4),
                    dbc.Col(children=right_bound,width=4)
                ],
                
            ),
            dbc.Row(
                children=[
                    dbc.Col(html.Div(),width=4,),
                    dbc.Col(children=bottom_bound,width=4),
                    dbc.Col(html.Div(),width=4,)
                ],
                
            )
        ]
    )

    ## Stack all previous divs together

    final_div = html.Div(
        children=[
            input_div,
            navigation_progress,
            load_button,
            grid_vals
        ],
        id="input_content_div"
    )
    return final_div


def get_segment_tab() -> html.Div:
    '''
    For the tab division, create the Segment tab separately.
    This will include all front-end elements needed for segmentation
    '''
    ## Make Segment and Markers visual boxes
    threshold_div = html.Div([
        html.Div("Threshold",id="threshold_image_label"),
        html.Div(id="threshold_image_div",
            className="segment_tab_image"
        )
    ]
    )
    markers_div = html.Div([
        html.Div("Markers",id="markers_image_label"),
        html.Div(id="markers_image_div",
            className="segment_tab_image"
        )
    ]
    )
    image_row = dbc.Row(
        [
            dbc.Col(threshold_div),
            dbc.Col(markers_div)
        ],
        id="image_row"
    )

    ## Get the vertical groupings of label and drop-down menu

    segmentation_methods = get_vertical_dropdown("Segmentation Method","segmentation_dropdown",SEGMENT_METHOD_DROPDOWN_OPTIONS)
    edge_detection_methods = get_vertical_dropdown("Edge Detection Method","edge_detection_dropdown",EDGE_DETECTION_DROPDOWN_OPTIONS)

    dropdown_row = dbc.Row(
        children=[
            dbc.Col(segmentation_methods),
            dbc.Col(edge_detection_methods)
        ],
        id="dropdown_row"
    )

    ## (WIP) Save and Load Buttons
    run_button = html.Div(
        children=html.Button(
            'Run',
            id="run_button"
        ),
    )
    save_h5_button = html.Div(
        children=[
            html.Button(
                'Save .h5...',
                id="save_h5_button",
            ),
            dcc.Download(id="save_h5_download")
        ]
    )

    segment_button_row = dbc.Row(
        children=[
            dbc.Col(run_button),
            dbc.Col(save_h5_button)
        ],
        id="segment_button_row"
    )

    ## Combine All
    total_div = html.Div(
        [
            image_row,
            dropdown_row,
            segment_button_row
        ],
 
    )
    return total_div

def get_label_tab() -> html.Div:
    '''
    Develop the tab that includes navigation of segmented images

    '''
    
    # Develop button presses for navigating labeling
    labeling_left_arrow = save_h5_button = html.Div(
        id="labeling_left_arrow",
        children=html.Button('<')
    )
    labeling_right_arrow = save_h5_button = html.Div(
        id="labeling_right_arrow",
        children=html.Button('>')
    )
    labeling_div = html.Div([
        html.Div("Labeling",id="labeling_image_label"),
        html.Div(id="labeling_image_div")
    ]
    )
    labeling_row = dbc.Row([
        dbc.Col(labeling_left_arrow),
        dbc.Col(labeling_div),
        dbc.Col(labeling_right_arrow)
    ])

    ## Develop Labeling Guide
    labeling_guide_title = html.Label("Labeling Guide",id="labeling_guide_title")
    labeling_instructions_div = html.Div("Waiting for input image",id="labeling_instructions_div")
    labeling_field = html.Div(
            dcc.Input(
                id="labeling_field",
                type='text',
                placeholder=""
            )
        ),
    labeling_column = dbc.Col([
        dbc.Row(labeling_guide_title),
        dbc.Row(labeling_instructions_div),
        dbc.Row(labeling_field)
    ])

    ## Save Button
    save_csv_button = html.Div(
        children=[
            html.Button(
                'Save Labeled .csv',
                id="save_csv_button"
            ),
            dcc.Download(id="save_csv_download")
        ]
    )

    ## Combine all divs
    return html.Div([
        labeling_row,
        labeling_column,
        save_csv_button
    ])


def get_tab_div() -> html.Div:
    '''
    Get the div associated with creating each of the tabs for Segmentation and Labeling in the dash applet
    '''
    total_tabs = dbc.Tabs([
        dbc.Tab(label="Segment",tab_id="segment_tab"),
        dbc.Tab(label="Label", tab_id="label_tab"),
    ],
    id="tabs_div",
    active_tab="segment_tab")
    tab_content_div = html.Div(id="tab_content_div")

    ## Create invisible 'memo' divs for holding 'children' 
    tab_segment_memo = html.Div(id="tab_segment_memo",
        style={"display":"none"},
        children=get_segment_tab()
    )
    tab_label_memo = html.Div(id="tab_label_memo",
        style={"display":"none"},
        children=get_label_tab()
    )

    tab_div = dbc.Col([
            dbc.Row(total_tabs),
            dbc.Row(tab_content_div),
            dbc.Row(tab_segment_memo),
            dbc.Row(tab_label_memo)
        ],
    )

    return tab_div

def get_main_div():
    '''
    Call this to get the organized divs for the entire segmentations applet. This is split as the
    Image load window and the segmentation walkthroughs
    '''
    image_load_div = get_image_load_div()

    tab_div = get_tab_div()

    return dbc.Row(children=[
        dbc.Col(image_load_div),
        dbc.Col(tab_div),
        html.Div(
            id="output_blank",
            style={"display":"none"}
        )
    ])

# ## Callbacks limited for Navigation and Loading ##

# @callback(
#     Output("tab_content_div","children"),
#     Input("tabs_div","active_tab")
# )
# def switch_tabs(tab_oi) -> html.Div:
#     if tab_oi == "segment_tab":
#         return get_segment_tab()
#     if tab_oi == "label_tab":
#         return get_label_tab()


if __name__ == "__main__":
    app = Dash(__file__,
    external_stylesheets=[dbc.themes.LUX])

    app.layout=get_main_div()

    app.run_server(port=8020,
    host='0.0.0.0')