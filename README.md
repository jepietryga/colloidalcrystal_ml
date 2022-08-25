# What is this?
This is a WIP tool for Image Segmenting colloidal crystal SEM Images and building models out of them to automate counting & classification.

Currently, Random Forest is the utilized model, but others may work!

# How to install
Use 
```
git clone https://github.com/jepietryga/colloidal_crystal_ML.git
```

or follow other methods with SSH or CLI!

After that, you may use the requirements.txt file inside a virtual environment to ensure you have the correct libraries.

The current Python version is 3.9.

# How to use
Notebooks is the main folder of interest. Here, several notebooks are available for segmenting and classifiying data, training and saving new models, and viewing data. Additionally, some experimental code is available for supplying additional features to data after the fact!

Finally, there is one notebook for actually running the currently made models against other images--using this requires user input in the notebooks to specify which images. Load images of interest into the `Images/Additional` directory.