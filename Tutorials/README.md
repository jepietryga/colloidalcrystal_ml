<h2>Terminal Commands</h2>
<ul>
<li><code>dash_applet</code><p>Running this command will open the Dash applet that connects to the interactive <code>facet_ml.segmentation.segmenter.ImageSegmenter</code>. Once an image is loaded, different segmentation styles and edge modifications can be used to identify image regions (Segment Tab), and then each of these identified regions can be labeled (Label Tab). The .h5 of the regions and labels with correspondin features can be saved as .h5 and .csv, respectively.</p> 
</li>
<li><code>apply_segmenter [-h] --image-path IMAGE_PATH --image-segmenter-kwargs IMAGE_SEGMENTER_KWARGS --output-path
OUTPUT_PATH</code>
<p>This callable simply applies an <code>ImageSegmenter</code> to get the dataframe of features. <code>IMAGE_PATH</code> is the path of the image to be loaded. <code>IMAGE_SEGMENTER_KWARGS</code> is a JSON-like string that contains keyword arguments for creating an <code>ImageSegmenter</code> object. <code>OUTPUT_PATH</code> specifies where the featurized regions are saved.</p>
</li>
<li><code>train_rf_model --data-path DATA_PATH --split-frac SPLIT_FRAC --output-path OUTPUT_PATH
[--feature-set FEATURE_SET] [--labels LABELS [LABELS ...]] [--model-params MODEL_PARAMS]
[--train-loops TRAIN_LOOPS]</code>
<p>This callable trains a RandomForestClassifier. <code>DATA_PATH</code> is a .csv of data features. <code>SPLIT_FRAC</code> is the fraction of test set data. <code>OUTPUT_PATH</code> is the save location of the RandomForestClassifier. <code>FEATURE_SET</code> defines the features to train on, associated with the lists in <code>facet_ml.classification.config_files.config_features.json</code>. <code>MODEL_PARAMS</code> are stored in  
<code>facet_ml.classification.config_files.config_model.json</code>. <code>LABELS</code> can optionally be specified which will ensure that only the requested labels are trained on. <code>TRAIN_LOOPS</code> defines how many iterations of RandomForest training are run to find hte best performing model based on F1-score metrics. </p>
</li>
<li><code>use_rf_model [-h] --data-path DATA_PATH --model-path MODEL_PATH --output-path OUTPUT_PATH</code>
<p>This callable uses a previously trained RandomForest model. <code>DATA_PATH</code> species the .csv to run through the RandomForest model. <code>MODEL_PATH</code> specifies the saved model to use. <code>OUTPUT_PATH</code> specifies where the labeled output .csv will be saved.</p>
</li>
</ul>

