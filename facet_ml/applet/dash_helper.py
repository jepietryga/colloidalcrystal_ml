# Helper functions for the dash app

from PIL import Image
import base64
from io import BytesIO
import numpy as np

def np_to_base64(img:np.array):
    '''
    Given a np array, create a base64 encoding
    '''
    pil = Image.fromarray(img.astype('uint8'))
    buffer = BytesIO()
    pil.save(buffer, format="PNG")

    val = buffer.getvalue()
    val_b64 = base64.b64encode(val).decode('utf-8')

    str_b64 = f"data:image/png;base64,{val_b64}"
    return str_b64

def upload_content_to_np(content):
    '''
    Given an content from dcc.Upload, convert it into a np.array
    '''
    content_type, content_string = content.split(",")

    decoded = base64.b64decode(content_string)

    image = Image.open(BytesIO(decoded))
    return np.array(image)

def region_roll(image_segmenter,region_val):
    '''
    Given a region, roll it if it exceeds max or min of regions
    '''
    if region_val > image_segmenter.df["Region"].max():
            return image_segmenter.df["Region"].min()
    elif region_val < image_segmenter.df["Region"].min():
        return image_segmenter.df["Region"].max()
    else:
        return region_val

def move_region(image_segmenter,val):
    '''
    Convenience function to adjust the image_segmenter's internal region_tracking
    Need to check if the region actually exists as some may disappear if neglibible size
    This moth-eaten region list will thus have skipped numbers
    '''
    check_val = 0 
    for ii in range(len(image_segmenter.df)):
        check_val += val
        region_oi = check_val+image_segmenter._region_tracker
        rolled_region = region_roll(image_segmenter, region_oi)
        region_valid = rolled_region in image_segmenter.df["Region"].to_list()
        if region_valid:
            image_segmenter._region_tracker = rolled_region
            return
    
    raise Exception("No valid regions found")

def parse_contents_img(contents, filename, date):
    '''
    Given input of an image file, return a np.ndarray rrepresenting the data

    '''
    image = Image.open(contents)
    return np.array(image)
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])