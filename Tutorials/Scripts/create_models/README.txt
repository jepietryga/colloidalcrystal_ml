There are several types of models used in this project.

<h3>Semantic</h3>
These algorithms or models are those that separate background from foreground, and are then passed to Watershed algorithm
<ul>
<li>Otsu</li>
<li>Ensemble</li>
<li>CNN</li>
</ul>

<h3>Instance</h3>
These algorithms or models are those that get distincxt objects, which can then be passed to the Classifier
<ul>
<li> Faster MaskRCNN (backgorund) </li>
<li> Segment Anything Model </li>
</ul>

<h3>Classifier</h3>
These algorithms or models are those that take identified regions and convert them into classes
<ul>
<li> CNN (classifier) </li>
<li> RandomForest </li>
<li> MultilayerRandomForest </li>
<ul>

<h3>Instance+Classifier</h3>
These algorithms or models are htose that take an image, detect objects, and then classify in one shot
<ul>
<li> MaskRCNN (full)</li>
<ul>