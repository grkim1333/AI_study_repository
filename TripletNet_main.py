from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from TripletNet_model import SingleLayer
from TripletNet_model import TripleLayer
from TripletNet_model import RandomInitNetwork
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train_all, x_test, y_train_all, y_test =train_test_split(x, y, stratify = y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val =  train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
print(len(x), len(x_train), len(x_val), len(x_test))

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled= scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)



random_init_net = RandomInitNetwork(units_1 = 10, units_2 = 5, l1 = 0.01, l2 = 0.01)
random_init_net.fit(x_train_scaled, y_train, x_val = x_val_scaled, y_val = y_val, epochs = 1000)
random_init_net.score(x_val_scaled,y_val)

plt.ylim(0, 0.7)
plt.plot(random_init_net.losses)
plt.plot(random_init_net.val_losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.title('Triple_Layer_Random_Init_Net_Plot')
plt.show()
