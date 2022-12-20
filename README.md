<h1>A simple implementation of Convolutional Neural Networks</h1>

<h2>Software</h2>

<div align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="python">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="MatPlotLib">

</div>

<div align="center">
    <img src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
    <img src="https://img.shields.io/badge/bootstrap-%23563D7C.svg?style=for-the-badge&logo=bootstrap&logoColor=white" alt="">
</div>

<p>
    The goal is to create a package that contains an automated implementation of a Convolutional Neural Network (CNN). 
    The model is going to take as input jpeg images, manipulate them and train on them in order to make predictions.
</p>

<h2>Image input module</h2>
<p>
    Loads and prepares images for training. The data-object should be optimal for use in the CNN. Images should also be 
    optimized for use when they are taken as input or at least there should be an option for it.
</p>

<h2>Neural Network</h2>
<p>
    This is a CNN which can take as input the image object and uses it to make predictions. The hyper-parameters should be 
    adjusted automatically in a hyper-parameter optimization step. Training should appear as a method in the object class and
    predictions should not contribute to the training of the dataset. The model should be able to produce some plots to 
    describe processes such as training or predicting.
</p>
