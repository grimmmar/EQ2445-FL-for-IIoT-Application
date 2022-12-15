# Federated Learning for Industrial Internet of Thing Application

EQ2445 - Project in Multimedia Processing and Analysis - KTH

# Install

Folder skeleton:

Federated-learning/

  - scr/
  
  - save/
  
  - data/
    
    - cifar
    
    - LEGO brick images v1
  
  - logs/
  
All the .py file should be put in the folder 'Federated-learning/src'

# Usage

- federated_main.py (Federated Learning)

- baseline_main.py (Centralized neuron network)

## Run the code locally
Run the cmd.exe under path 'Federated-learning/'

Command Example:

<python src/federated_main.py --model=cnn --dataset=LEGO --gpu=0 --epochs=30 --num_users=20 --selected_users=18 --snr_dB=10.0 --iid=0> 

## Run the code on google-colab
Create the folders in the same way as in local. Use [federated_colab.ipynb](https://github.com/grimmmar/EQ2445-FL-for-IIoT-Application/blob/main/federated_colab.ipynb) directly.

# Environment

- Python 3.9.6

- Pytorch 1.13.0+cu117

- Cuda 11.7

# Project Diary

Updated 03/12/2022

Add 'snr' in Options

Wireless simulation completed

Updated 04/12/2022

Use 'snr_dB' instead of 'snr'

Use step decay lr

Updated 07/12/2022

Modify the incorrect implementation of wireless part

Updated 08/12/2022

Modify the incorrect implementation of wireless part

Add the hard node selection part

Updated 12/11/2022

Simplify the code

Updated 13/11/2022

Add non-iid part for LEGO dataset

Updated 15/11/2022

Add MSE calculator

# Notice

[LEGO dataset](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images) has 16 classes. 

After unzipping the file, move the folder 'LEGO brick images v1' under path '\Federated Learning\data', where you can find the cifar folder and the fmnist folder.
