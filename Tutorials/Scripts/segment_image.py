from facet_ml.segmentation.segmenter import ImageSegmenter

#### Specify variables ####
save_id = "name" # Will save the h5 and csv

input_path = None
pixels_to_um = 9.81 # Converison between pixels and microns


segmenter = "algorithmic" # From "algorithmic","segment_anything", and (not general) "maskrcnn"
segmenter_kwargs = {
    "threshold_mode":"otsu",
    "edge_modification":None # "localthresh","canny","darkbright"
    }
#### End Specify Variables ####

#### Begin Code ####

IS = ImageSegmenter(input_path=input_path,
               segmenter=segmenter,
               segmenter_kwargs=segmenter_kwargs,
               )

IS.df.to_csv(f"{save_id}.csv")
IS.to_h5(f"{save_id}.h5","w")