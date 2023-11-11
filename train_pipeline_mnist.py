from torch.utils.data import DataLoader
from myDataset import myDataset

from lbae import LBAE
from pipeline import Pipeline
from rbm import RBM


NUM_VISIBLE = 64
MAX_EPOCHS = 100
NUM_HIDDEN = 64
RBM_STEPS = 1000
BATCH_SIZE = 8


train_dataset = myDataset(dataset_part='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

autoencoder = LBAE(input_size=(1, 8, 8), out_channels=8, zsize=NUM_VISIBLE, num_layers=2, quantize=list(range(MAX_EPOCHS)))
rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm, classifier=True)

pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS)