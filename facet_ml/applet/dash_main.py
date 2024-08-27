# Main Chunk of code for writing dash
from dash import Dash, dash_table, dcc, html, callback, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from PIL import Image
from io import BytesIO
import h5py

import numpy as np

from facet_ml.segmentation.segmenter import (
    BatchImageSegmenter,
    ImageSegmenter,
    AlgorithmicSegmenter,
    SAMSegmenter,
    MaskRCNNSegmenter,
)

from facet_ml.applet import dash_divs, dash_helper

## Mappers
segment_mode_mapper = {
    "Otsu (Global) Binarization": {
        "segmenter": AlgorithmicSegmenter,
        "segmenter_kwargs": {"threshold_mode": "otsu"},
    },
    "Local Threshold": {
        "segmenter": AlgorithmicSegmenter,
        "segmenter_kwargs": {"threshold_mode": "local"},
    },
    "Pixel Classifier": {
        "segmenter": AlgorithmicSegmenter,
        "segmenter_kwargs": {"threshold_mode": "pixel"},
    },
    "Ensemble": {
        "segmenter": AlgorithmicSegmenter,
        "segmenter_kwargs": {"threshold_mode": "ensemble"},
    },
    "Detectron2": {
        "segmenter": MaskRCNNSegmenter,
        "segmenter_kwargs": {},
    },
    "Segment Anything": {
        "segmenter": SAMSegmenter,
        "segmenter_kwargs": {
            "sam_kwargs": {"points_per_side": 64},
        },
    },
}

edge_mode_mapper = {
    "Local Thresholding": "localthresh",
    "Bright-Dark": "darkbright",
    "None": None,
    "Testing": "testing",
}

# Create App
app = Dash(__file__, external_stylesheets=[dbc.themes.LUX])
app.layout = dash_divs.get_main_div()


# Create simple state tracking class
class AppState:

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

        # Initialize Tabs information
        self.tabs = {}

        self.tabs["segment"] = dash_divs.get_segment_tab()
        self.tabs["label"] = dash_divs.get_label_tab()
        self.active_tab = "segment"

    @property
    def image_segmenter(self):
        return self.batch_image_segmenter[self.batch_tracker]


## Callbacks limited for Navigation and Loading ##


@callback(
    # Output("tab_content_div","children"),
    Output("tab_segment_memo", "style"),
    Output("tab_label_memo", "style"),
    Input("tabs_div", "active_tab"),
    # State("tab_segment_memo","children"),
    # State("tab_label_memo","children"),
)
def switch_tabs(
    tab_oi,
) -> html.Div:
    if tab_oi == "segment_tab":
        return {"display": "block"}, {"display": "none"}
    if tab_oi == "label_tab":
        return {"display": "none"}, {"display": "block"}


## Wire functionality ##

# State tracker
state = AppState()
batch_image_segmenter = None
image_segmenter = None
images = None
filenames = None

# State groups, use Dash flexible callbacks
image_inputs_state = {
    "top_bound": State("top_bound_field", "value"),
    "bottom_bound": State("bottom_bound_field", "value"),
    "left_bound": State("left_bound_field", "value"),
    "right_bound": State("right_bound_field", "value"),
    "px_to_um": State("px_to_um_field", "value"),
}
segmentation_inputs_state = {
    "segmentation_mode": State("segmentation_dropdown", "value"),
    "edge_detection_mode": State("edge_detection_dropdown", "value"),
}


@callback(
    output=[Output("input_image_div", "children")],
    inputs=dict(list_contents=Input("load_button", "contents")),
    state=dict(list_filenames=State("load_button", "filename")),
    prevent_initial_call=True,
)
def get_input_file(list_contents, list_filenames):
    """
    When load_button pressed, load a file. Store the image data and filenames,
    and also load in the image
    """

    # Load the image(s) (WIP)
    # NOTE: May need to temp save these
    from io import BytesIO
    import copy

    # state.images = [np.array(Image.open(BytesIO(content))) for content in list_contents]
    state.images = [
        dash_helper.upload_content_to_np((content)) for content in list_contents
    ]
    state.filenames = list_filenames

    return [html.Img(src=list_contents[0], className="processed_image")]
    # raise NotImplemented


@callback(
    output=[
        Output("threshold_image_div", "children"),
        Output("markers_image_div", "children"),
    ],
    inputs=dict(
        run_click=Input("run_button", "n_clicks"),
    ),
    state=image_inputs_state | segmentation_inputs_state,
    prevent_initial_call=True,
    suppress_callback_exceptions=True,
)
def perform_segmentation(
    run_click,
    top_bound,
    bottom_bound,
    left_bound,
    right_bound,
    px_to_um,
    segmentation_mode,
    edge_detection_mode,
):
    # Edge Case: Do not run if no images
    if len(state.images) == 0:
        return ([html.Div(className="blank_image")], [html.Div("blank_image")])
    # Segment Mode mapper
    segmenter_instructions = segment_mode_mapper[segmentation_mode]
    edge_arg = edge_mode_mapper[edge_detection_mode]
    if isinstance(segmenter_instructions["segmenter"], AlgorithmicSegmenter):
        segmenter_instructions["segmenter_kwargs"]["edge_modification"] = edge_arg

    # Initialize batch segmenter and control state
    state.batch_image_segmenter = BatchImageSegmenter(
        img_list=state.images,
        filename_list=state.filenames,
        top_boundary=int(top_bound) if top_bound else 0,
        bottom_boundary=int(bottom_bound) if bottom_bound else 860,
        left_boundary=int(left_bound) if left_bound else 0,
        right_boundary=int(right_bound) if right_bound else 2560,
        pixels_to_um=float(px_to_um) if px_to_um else 9.37,
        segmenter=segmenter_instructions["segmenter"],
        segmenter_kwargs=segmenter_instructions["segmenter_kwargs"],
    )
    state.batch_tracker = 0

    # Run the segmentation
    state.batch_image_segmenter.df

    # Return modifications to the visualized images
    thresh_b64 = dash_helper.np_to_base64(state.image_segmenter.thresh)
    markers_b64 = dash_helper.np_to_base64(state.image_segmenter.markers_filled)
    thresh_div = [html.Img(src=thresh_b64, className="processed_image")]
    markers_div = [html.Img(src=markers_b64, className="processed_image")]
    return (thresh_div, markers_div)


@callback(
    output=Output("save_h5_download", "data"),
    inputs=Input("save_h5_button", "n_clicks"),
    prevent_initial_call=True,
)
def save_segmentation(n_clicks):
    """
    Create, send, and locally delete the h5 file.
    """
    buffer = BytesIO()
    f = h5py.File(buffer, "w")
    f.close()
    for IS in state.batch_image_segmenter.IS_list:
        IS.to_h5(buffer, "r+")

    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "segmented.h5")


@callback(
    output=[
        Output("input_image_div", "children", allow_duplicate=True),
        Output("threshold_image_div", "children", allow_duplicate=True),
        Output("markers_image_div", "children", allow_duplicate=True),
    ],
    inputs=[
        Input("input_left_arrow", "n_clicks"),
        Input("input_right_arrow", "n_clicks"),
    ],
    state=[
        State("input_image_div", "children"),
        State("threshold_image_div", "children"),
        State("markers_image_div", "children"),
    ],
    prevent_initial_call=True,
)
def input_arrows(
    left_clicks,
    right_clicks,
    iid,
    tid,
    mid,
):
    """
    Shift displayed image right or left
    """
    if len(state.images) == 0:
        return [iid, tid, mid]

    triggered_id = ctx.triggered_id
    if triggered_id == "input_right_arrow":
        state.batch_tracker += 1
    elif triggered_id == "input_left_arrow":
        state.batch_tracker -= 1
    state.batch_tracker = state.batch_tracker % len(state.images)

    # Make new image
    input_b64 = dash_helper.np_to_base64(state.image_segmenter.image_read)
    thresh_b64 = dash_helper.np_to_base64(state.image_segmenter.thresh)
    markers_b64 = dash_helper.np_to_base64(state.image_segmenter.markers)
    return [
        [html.Img(src=input_b64, className="processed_image")],
        [html.Img(src=thresh_b64, className="processed_image")],
        [html.Img(src=markers_b64, className="processed_image")],
    ]


@callback(
    output=[
        Output("labeling_instructions_div", "children"),
        Output("labeling_image_div", "children", allow_duplicate=True),
        Output("labeling_field", "value"),
    ],
    inputs=[
        Input("labeling_left_arrow", "n_clicks"),
        Input("labeling_right_arrow", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def labeling_arrows(left_click, right_click):
    if len(state.images) == 0:
        return [iid, tid, mid]

    triggered_id = ctx.triggered_id
    if triggered_id == "labeling_right_arrow":
        dash_helper.move_region(state.image_segmenter, 1)
    elif triggered_id == "labeling_left_arrow":
        dash_helper.move_region(state.image_segmenter, -1)

    # Provide update to grab the region of interest
    text_update = f"""
        Write label for Region {state.image_segmenter._region_tracker} below (Total Regions: {len(state.image_segmenter.df)})\n
    """
    region_target = state.image_segmenter.region_dict[
        state.image_segmenter._region_tracker
    ]
    image_64 = dash_helper.np_to_base64(region_target)
    image_div = html.Img(src=image_64)

    # Also modify the label field
    label_text = state.image_segmenter.df.loc[
        state.image_segmenter.df["Region"] == state.image_segmenter._region_tracker,
        "Labels",
    ].tolist()[0]
    if label_text is None:
        label_text = ""

    return [text_update, [image_div], label_text]


@callback(
    output=Output("output_blank", "children"), inputs=Input("labeling_field", "value")
)
def update_df_region_label(val):
    if state.batch_image_segmenter:
        state.image_segmenter.update_df_label_at_region(val)

    return [html.Div()]


@callback(
    output=Output("save_csv_download", "data"),
    inputs=Input("save_csv_button", "n_clicks"),
    prevent_initial_call=True,
)
def save_label(n_clicks):
    return dcc.send_data_frame(state.batch_image_segmenter.df.to_csv, "labeled.csv")
    raise NotImplemented


def run_app():
    """
    Simple script for running the dash_app
    """
    app.run_server(host="0.0.0.0", port=8050, debug=True)


if __name__ == "__main__":
    run_app()
