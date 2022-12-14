import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
'''
path = r'save'
data1 = pd.read_csv('../save/LEGO_cnn_30.csv')

y1_l = data1.loc[:, 'loss_no']
y1_a = data1.loc[:, 'accuracy_no']
y2_l = data1.loc[:, 'loss_10']
y2_a = data1.loc[:, 'accuracy_10']
y3_l = data1.loc[:, 'loss_0']
y3_a = data1.loc[:, 'accuracy_0']

plt.figure(1)
plt.plot(range(len(y1_l)), y1_l, color='r', label='noiseless', linewidth=1)
plt.plot(range(len(y2_l)), y2_l, color='b', label='SNR=10dB', linewidth=1)
plt.plot(range(len(y3_l)), y3_l, color='y', label='SNR=0dB', linewidth=1)
plt.title('Training Loss Using LEGO Dataset')
plt.xlabel('FL training round')
plt.ylabel('Training loss')
plt.legend()

plt.figure(2)
plt.plot(range(len(y1_a)), y1_a, color='r', label='noiseless', linewidth=1)
plt.plot(range(len(y2_a)), y2_a, color='b', label='SNR=10dB', linewidth=1)
plt.plot(range(len(y3_a)), y3_a, color='y', label='SNR=0dB', linewidth=1)
plt.title('Training Accuracy Using LEGO Dataset')
plt.xlabel('FL training round')
plt.ylabel('Training accuracy')
plt.legend()
plt.show()
'''

x = np.linspace(10, 20, 6)
y_1 = [95.37, 94.81, 95.61, 94.67, 95.22, 94.59]
y_2 = [90.04, 90.12, 89.1, 88.31, 88.16, 80.55]
y_3 = [88.24, 92.24, 93.18, 93.25, 93.18, 92.78]
y_4 = [83.69, 86.12, 88.94, 91.06, 82.12]
y_1 = [y_1[i] / 100 for i in range(6)]
y_2 = [y_2[i] / 100 for i in range(6)]
y_3 = [y_3[i] / 100 for i in range(6)]
y_4 = [y_4[i] / 100 for i in range(5)]
plt.plot(x, y_1, color='r', linewidth=1, label='SNR=10dB iid', marker='o', markerfacecolor='white')
plt.plot(x, y_2, color='b', linewidth=1, label='SNR=0dB iid', marker='o', markerfacecolor='white')
plt.plot(x, y_3, color='y', linewidth=1, label='SNR=10dB non-iid', marker='o', markerfacecolor='white')
plt.plot(x[:5], y_4, color='g', linewidth=1, label='SNR=0dB non-iid', marker='o', markerfacecolor='white')
plt.title('Hard node selection')
plt.xlabel('the number of selected users')
plt.ylabel('Test Accuracy')
plt.ylim([0.5, 1])
plt.legend()
plt.show()
