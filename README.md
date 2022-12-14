<h1>A simple implementation of Convolutional Neural Networks</h1>

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.9.12-blue" alt="Python v3.9.12">
    <img src="https://img.shields.io/badge/PyTorch-1.10.2-blue" alt="Pytorch v1.10.2">
    <img src="https://img.shields.io/badge/NumPy-1.23.3-blue" alt="NumPy">
    <img src="https://img.shields.io/badge/MatPlotLib-3.5.2-blue" alt="MatPlotLib">
</div>

<h2>Description</h2>
<p>
    The goal is to create a package that contains an automated implementation of a Convolutional Neural Network (CNN). 
    The package will be accompanied by a script (<em>main.py</em>) which consists of an implementation of the package.
    The model is going to take as input jpeg images, manipulate them and train on them in order to make predictions.
</p>

<h2>Image input module</h2>
<p>
    Loads and prepares images for training. The data-object should be optimal for use in the CNN. Images should also be 
    optimized for use when they are taken as input or at least there should be an option for it.
</p>
<p>
    Images should be placed in a proper file structure in order for them to be loaded properly through PyTorch.
</p>

<h2>Neural Network</h2>
<p>
    This is a CNN which can take as input the image object and uses it to make predictions. The hyper-parameters should be 
    adjusted automatically in a hyper-parameter optimization step. Training should appear as a method in the object class and
    predictions should not contribute to the training of the dataset. The model should be able to produce some plots to 
    describe processes such as training or predicting.
</p>

<h2>Report</h2>
<p>
    At the end of running the pipeline should be able to produce a report in the form of HTML files.
    The report is split in three units (tabs): Model training information, Prediction infomration and Plots.
    Each of those units is only created if the step has been used. E.g. in order to get information in the prediction tab 
    the model should be used to make some predictions.
</p>