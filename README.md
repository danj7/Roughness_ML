# Roughness_ML
This code is a 'test of concept' to find out if Machine Learning can be used to classify images of interfaces according to their *roughness exponent* (To-do: link to explanation). This is an improved version of a project I did in 2018 during a course on parallel programming on GPUs and is based on research done for a Master's Thesis in Condensed Matter Physics at the Instituto Balseiro, in Argentina.

This code creates a model from artificial interfaces created using an algorithm learned from Dr. Alejandro Kolton (To-do: link to explanation) and is assuming binarized images of interfaces like the following:

[image]

## Setting up
The model is constructed using Keras with the TensorFlow backend. I am also using the Anaconda Python 3.7 distribution, a GeForce GTX 960M graphics card, and I am running Manjaro Linux . Installing all the components needed was a little tricky but one should probably start by installing the NVIDIA Drivers from https://www.nvidia.com/Download/index.aspx (I installed the latest one, version 430.34) and the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads (I installed version 10.1, but that caused some problems I'll explain later on).

Then one needs to install TensorFlow. I did it as the [website](https://www.tensorflow.org/install) suggested, using
```
pip install tensorflow-gpu
```
since I planned to use my GPU. But, I had problems getting my code to use my GPU with CUDA Toolkit version 10.1 so I installed version 10.0 and cuDNN using `conda`:
```
$ conda install cudatoolkit=10.0.130
$ conda install cudnn
```

Finally, I installed [Keras](https://keras.io/) using:
```
$ sudo pip install keras
```
and all seems to work.

## Usage
Use the included Jupyter Notebook which includes all the necessary functions to work with.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
* Since this problem seemed to me to be similar to character recognition, I based a lot of the code from the [MNIST CNN example](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
* I also learned a lot of what this all means from a [blog post](https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9) on image classification by Rising Odewa
* Once again, thanks to [Dr. Alejandro Kolton](http://cabtes55.cnea.gov.ar/solidos/personales/kolton/) for teaching me how to generate this interfaces.
