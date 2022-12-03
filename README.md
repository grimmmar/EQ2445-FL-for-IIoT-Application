Notice: the LEGO dataset has 16 classes. And since its dataset contains less images than cifar10 or fmnist, the number of users need to be small, or there will be an error.

The dataset can be found in https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images

After unzipping the file, move the folder 'LEGO brick images v1' under path '\Federated Learning\data', where you can find the cifar folder and the fmnist folder.

Updated 03/12/2022

Add 'snr' in Options

!python src/federated_main.py --model=cnn --dataset=LEGO --gpu=0 --iid=1 --epochs=20 --local_bs=64 --num_users=10 --local_ep=5 --snr=3.0

Wireless simulation completed
