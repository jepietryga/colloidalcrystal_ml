# What is this?
This is a tool for ImageSegmenting colloidal crystal SEM Images and building models out of them to automate counting & classification.

The main use of this repository is to access the ImageSegmenter class as it allows users to utilize different computer vision tools through classes and applets forp erfomring segmentation and labeling.

Additionally, code for helping train and utilize RandomForest classification models is also included.

# How to install

<ol>
<li> Get the repository using <code>git clone https://github.com/jepietryga/colloidal_crystal_ML.git</code>. </li>
<li> Install dependencies using <code>conda env create -f environment.yml</code>.</li>
<li> Install the library using <code>pip install -e ./</code>. The editable version is the easiest to move models into</li>
<li> Navigate to <code>./facet_ml/static/Models</code> and follow instructions in its README.md to download models from Zenodo.
</ol>

# How to use

`facet_ml` holds the main body of code, which includes code for segmenters, models, and utility scripts for handling images as well as the applet code.

`Tutorials` holds notebook and script files that walkthrough usage of some of the code. This includes segmentation, classifier training, and data viewing.

To quickly access the assisted labeling applet, just write `dash_applet` in the terminal and follow its instructions.

# Data and Models
Processed data and models are shared via <a href="https://doi.org/10.5281/zenodo.14019586">Zenodo</a>. There is room for further optimization of models, but those used in the linked paper's data analysis are kept for reproducibility and transparency.

# Citing
If you found this codebase to be useful, it is helpful to cite the paper!
This code is being developed for a yet unpublished paper. Information will be attached at a later date.

# Issues and Additions
The codebase is developing as research needs change. Please raise an issue if something is not working as intended, and feel free to recommended new methods!