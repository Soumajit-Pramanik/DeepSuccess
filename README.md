# DeepSuccess
This repository contains the codes to run the DeepSuccess model for predicting group success and the group profile components in future.

## Setup
  + We require python 2.7 and a handful of other supporting libraries for running these codes. 
  + Please create a python environment using conda via the following command:
    
    ```conda create -y -n deepsuccess27 python=2.7```
  + Activate the environment:
    
    ```conda activate deepsuccess27```
  + Install pytorch and numpy:
    
    ```python2.7 -m pip install torch```
    
    ```python2.7 -m pip install numpy```

## Dataset
  + A small "Toy" dataset is provided in the google drive location https://drive.google.com/drive/folders/1fUdvv7NgeY2YBaEcO3Kb4ICVdoEP7uv9?usp=sharing for testing the provided codes.
  + Please download the "Toy_Dataset" folder and keep it in the same folder as the codes present here.

## How to run?
  + Please run the deepsuccess.py using the following command:
    
    ```python deepsuccess.py```
  + It trains the model and provide training and testing accuracies for group success as well as group profile components.

## Contact
  + For any query, please contact Soumajit Pramanik at soumajit[dot]pramanik[at]gmail[dot]com .
