<h2> Downloading Files </h2>

<p>After installing the package, run <code>download_models</code> to fetch the Zenodo-hosted models and the Segment Anything checkpoint automatically.</p>

<p>The downloader uses the Zenodo record at <a href="https://doi.org/10.5281/zenodo.14019586">10.5281/zenodo.14019586</a> and places known files into the locations expected by <code>facet_ml</code>.</p>

<li> Default download command: <code>download_models</code> </li>
<li> Re-download and replace existing files: <code>download_models --overwrite</code></li>
<li> Skip the Segment Anything checkpoint: <code>download_models --skip-sam</code></li>
</ul>
<p>The mask_rcnn model needs to be placed in a folder called <code>torch</code> to be utilized.</p>

</ul>

<h2> Small Notes </h2>
<p> It is noted that the Background Pixel Classifier model is quite large. This is due to the depth of the trees (averaging 57) in the RandomForestClassifer. For reproducibility, the model is shared as is, but it is recommended to focus on optimization of models in future work. </p>
