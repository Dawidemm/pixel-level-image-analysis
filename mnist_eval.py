import torch
import lightning as pl
from myDataset import myDataset
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance

from lbae import LBAE
# if torch.cuda.is_available():
model = LBAE.load_from_checkpoint(checkpoint_path='E:/projects/pixel-level-image-analysis/lightning_logs/version_12/checkpoints/epoch=99-step=18000.ckpt', 
                                    hparams_file='E:/projects/pixel-level-image-analysis/lightning_logs/version_12/hparams.yaml',
                                    map_location=torch.device('cpu'))
    
# else:
#     model = LBAE.load_from_checkpoint(checkpoint_path='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_3/checkpoints/epoch=19-step=28740.ckpt', 
#                                     hparams_file='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_3/hparams.yaml',
#                                     map_location=torch.device('cpu'))


test_dataset = myDataset(dataset_part='test')
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

X = torch.Tensor(test_dataset.dataset_data/16).reshape(len(test_dataset.dataset_data), 1, 8, 8)
preds = model(X)

X_test = X.clone().detach().reshape(360, 64)
preds = preds.clone().detach().reshape(360, 64)

print(torch.mean(pairwise_euclidean_distance(X_test, preds)))

print(X[0])
print(preds[0])