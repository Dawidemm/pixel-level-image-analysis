from sklearn import datasets
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lbae import LBAE
from pipeline import Pipeline
from rbm import RBM

digits = datasets.load_digits()

dataset_split = int(len(digits.data) * 0.8)

X_train, X_test = digits.data[:dataset_split]/16, digits.data[dataset_split:]/16
y_train, y_test = digits.target[:dataset_split], digits.target[dataset_split:]

X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

train_dataloader = DataLoader(list(zip(X_train,y_train)), batch_size=1, shuffle=False)

NUM_VISIBLE = 60
MAX_EPOCHS = 35
NUM_HIDDEN = 40
RBM_STEPS = 1000

autoencoder = LBAE(input_size=(1, 8, 8), out_channels=8, zsize=NUM_VISIBLE, num_layers=2, quantize=list(range(MAX_EPOCHS)))
rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm, classifier=True)

pipeline.fit(train_dataloader,  rbm_steps=RBM_STEPS)