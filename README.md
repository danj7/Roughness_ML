# Roughness_ML
This code is a 'test of concept' to find out if Machine Learning can be used to classify images of interfaces according to their *roughness exponent* (To-do: link to explanation). This is an improved version of a project I did in 2018 during a course on parallel programming on GPUs and is based on research done for a Master's Thesis in Condensed Matter Physics at the Instituto Balseiro, in Argentina.

This code creates a model from artificial interfaces created using an algorithm learned from Dr. Alejandro Kolton (To-do: link to explanation of algorithm) and is assuming binarized images of interfaces. To obtain a binarized image in a laboratory setting one starts with an image like the following:
![Complete image](https://user-images.githubusercontent.com/13749006/63537806-a321ce00-c4e4-11e9-8456-c38ae05b97bb.png)

Then, one can take a section from the original:

![Partial image](https://user-images.githubusercontent.com/13749006/63538057-1deae900-c4e5-11e9-98a6-57366b5e68e4.png)

and then binarized. In this case, I also rotated it:

![Binarized image](https://user-images.githubusercontent.com/13749006/63538100-30652280-c4e5-11e9-8433-b560fb0940be.png)

Sample artificial interfaces will be shown in the Jupyter Notebook.

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
You can clone the repo to a folder using
```
$ git clone https://github.com/danj7/Roughness_ML.git
```
and in the included Jupyter Notebook is the code that loads all the packages needed, builds the model and shows sample interfaces. It also builds a random testing set of interfaces to predict values for. However, this can slow down the browser (I am using Firefox) a lot, so the other option is to run the Python 3 file `roughness_ml.py` from the terminal,
```
$ python roughness_ml.py
```
and it will do the same thing, as well as write a file called `model_results.txt` which I used to test different versions of the model.

## Results
Training and validating is done with 2000 interfaces per zeta value over 10 epochs. The resulting model has a validation accuracy of 98.1%. However, seeing how there are so many classes in this exercise (20, though in real life there is an infinite number of *roughness exponents*) and the details/features that the model has to take into consideration are a lot, every prediction made by the model is always off. I calculated an approximate *rms error* of 0.2 for any prediction.


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
* Since this problem seemed to me to be similar to character recognition, I based a lot of the code from the [MNIST CNN example](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
* I also learned a lot of what this all means from a [blog post](https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9) on image classification by Rising Odewa
* Once again, thanks to [Dr. Alejandro Kolton](http://cabtes55.cnea.gov.ar/solidos/personales/kolton/) for teaching me how to generate this interfaces.
