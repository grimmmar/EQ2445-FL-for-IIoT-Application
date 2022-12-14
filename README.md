# Federated Learning for Industrial Internet of Thing Application

EQ2445 - Project in Multimedia Processing and Analysis - KTH

Project Owner: Deyou Zhang

Main contributor: Hanqi Yang (wireless simulation), ![Rongfei Pan](https://github.com/Tim-RongfeiPan) (neuron network construction)

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

## Run the code locally
Run the cmd.exe under path 'Federated-learning/'

Command Example:

<python src/federated_main.py --model=cnn --dataset=LEGO --gpu=0 --epochs=30 --num_users=20 --selected_users=18 --snr_dB=10.0 --iid=0> 

## Run the code on google-colab
Create the folders in the same way as in local. Use the colab file ![federated_colab.ipynb](https://github.com/grimmmar/EQ2445-FL-for-IIoT-Application/blob/main/federated_colab.ipynb) directly.

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

# Notice

the LEGO dataset has 16 classes. And since its dataset contains less images than cifar10 or fmnist, the number of users need to be small, or there will be an error.

The dataset can be found in https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images

After unzipping the file, move the folder 'LEGO brick images v1' under path '\Federated Learning\data', where you can find the cifar folder and the fmnist folder.
