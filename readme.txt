FL-CNN-MNIST: !python src/federated_main.py --model=cnn --dataset=fmnist --gpu=0 --iid=1 --epochs=20 --local_bs=64
FL-VGG16-CIFAR10 !python src/federated_main.py --model=vgg --dataset=cifar --gpu=0 --iid=1 --epochs=350 --local_bs=64 --num_channels=3
CentralizedNN-VGG16-CIFAR10 !python src/baseline_main.py --model=vgg --dataset=cifar --gpu=0 --epochs=100 --local_bs=64 --num_channels=3
