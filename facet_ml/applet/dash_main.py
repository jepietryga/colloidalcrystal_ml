# Main Chunk of code for writing dash
from dash import Dash, dash_table, dcc, html, callback, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from PIL import Image

import numpy as np

from facet_ml.segmentation.segmenter import (
    BatchImageSegmenter, 
    ImageSegmenter,
    AlgorithmicSegmenter,
    SAMSegmenter,
    MaskRCNNSegmenter
)

from facet_ml.applet import dash_divs, dash_helper

## Mappers
segment_mode_mapper = {
    "Otsu (Global) Binarization":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"otsu"},
                                },
    "Local Threshold":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"local"},
                                },
    "Pixel Classifier":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"pixel"},
                                },
    "Ensemble":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"ensemble"},
                                },
    "Detectron2":{"segmenter":MaskRCNNSegmenter,
                                  "segmenter_kwargs":{},
                                },
    "Segment Anything":{"segmenter":SAMSegmenter,
                                  "segmenter_kwargs":{
                                      "sam_kwargs":{"points_per_side":64},
                                  },
                                },
}

edge_mode_mapper = {
    "Local Thresholding":"localthresh",
    "Bright-Dark":"darkbright",
    "None":None,
    "Testing":"testing"
}

# Create App
app = Dash(__file__,
    external_stylesheets=[dbc.themes.LUX]
)
app.layout = dash_divs.get_main_div()

# Create simple state tracking class
class AppState():

    def __init__(self):

        # Read-ins
        self.images = []
        self.filenames = []

        # Check cv images
        self.batch_image_segmenter = None
        self.batch_tracker = None
        # self.image_segmenter = None
        self.cv_image = None
        self.file_path = None

        # Image Reading varibales
        self.top_boundary = None
        self.right_boundary = None
        self.bottom_boundary = None
        self.left_boundary = None
        self.pixels_to_um = None

    @property
    def image_segmenter(self):
        return self.batch_image_segmenter[self.batch_tracker]

## Wire functionality ##

# State tracker
state = AppState()
batch_image_segmenter = None
image_segmenter = None
images = None
filenames = None

# State groups, use Dash flexible callbacks
image_inputs_state = {
    "top_bound":State("top_bound_field","value"),
    "bottom_bound":State("bottom_bound_field","value"),
    "left_bound":State("left_bound_field","value"),
    "right_bound":State("right_bound_field","value"),
    "px_to_um":State("px_to_um_field","value")
}
segmentation_inputs_state = {
    "segmentation_mode":State("segmentation_dropdown","value"),
    "edge_detection_mode":State("edge_detection_dropdown","value")
}

@callback(
    output=[Output("input_image_div","children")],
    inputs=dict(
        list_contents=Input("load_button","contents")
    ),
    state=dict(
        list_filenames=State("load_button","filenames")
    ),
    prevent_initial_call=True
)
def get_input_file(list_contents,list_filenames):
    '''
    When load_button pressed, load a file. Store the image data and filenames,
    and also load in the image
    '''

    # Load the image(s) (WIP)
    # NOTE: May need to temp save these
    from io import BytesIO
    import copy
    # state.images = [np.array(Image.open(BytesIO(content))) for content in list_contents]
    state.images = [dash_helper.upload_content_to_np((content)) 
        for content in list_contents
    ]
    
    state.filenames = list_filenames

    return [html.Img(src=list_contents[0])]
    # raise NotImplemented

@callback(
    output=[
            Output("threshold_image_div","children"),
            Output("markers_image_div","children")
        ],
    inputs=dict(
        run_click=Input("run_button","n_clicks"),
    ),
    state=image_inputs_state | segmentation_inputs_state,
    prevent_initial_update=True,
    suppress_callback_exceptions=True
)
def perform_segmentation(
    run_click,
    top_bound,bottom_bound,left_bound,right_bound,px_to_um,
    segmentation_mode,edge_detection_mode
):
    print("BUTTON CLICK")
    # Edge Case: Do not run if no images
    if len(state.images) == 0:
        print("No images")
        return ([html.Div()],[html.Div()])
    # Segment Mode mapper
    segmenter_instructions = segment_mode_mapper[segmentation_mode]
    edge_arg = edge_mode_mapper[edge_detection_mode]
    if isinstance(segmenter_instructions["segmenter"],AlgorithmicSegmenter):
        segmenter_instructions["segmenter_kwargs"]["edge_modification"] = edge_arg 

    # Initialize batch segmenter and control state
    state.batch_image_segmenter = BatchImageSegmenter(
        img_list=state.images,
        top_boundary=top_bound or 0,
        bottom_boundary=bottom_bound or 860,
        left_boundary=left_bound or 0,
        right_boundary=right_bound or 2560,
        pixels_to_um=px_to_um or 9.37,
        segmenter=segmenter_instructions["segmenter"],
        segmenter_kwargs=segmenter_instructions["segmenter_kwargs"],
    )
    state.batch_tracker=0
    print("EBatch created")

    # Run the segmentation
    state.batch_image_segmenter.df

    # Return modifications to the visualized images
    thresh_b64 = dash_helper.np_to_base64(state.image_segmenter.thresh)
    markers_b64 = dash_helper.np_to_base64(state.image_segmenter.markers_filled)
    
    print("End of perform_segmentation")
    return ([html.Img(thresh_b64)], [html.Img(markers_b64)])

def save_segmentation():
    raise NotImplemented

def input_right():
    raise NotImplemented

def input_right():
    raise NotImplemented

def update_image_reading():
    '''
    Impact how the image will be read in by the applet,
    adjust visualization of leftmost image
    '''
    raise NotImplemented

def forward_label_click():
    raise NotImplemented

def back_label_click():
    raise NotImplemented

def update_df_region_label():
    raise NotImplemented

def save_label():
    raise NotImplemented




app.run_server(host="0.0.0.0",
port=8050,
debug=True
)