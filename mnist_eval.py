import torch
import lightning as pl
from myDataset import myDataset
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
import matplotlib.pyplot as plt

from lbae import LBAE
if torch.cuda.is_available():
    model = LBAE.load_from_checkpoint(checkpoint_path='E:/projects/pixel-level-image-analysis/lightning_logs/version_0/checkpoints/epoch=99-step=18000.ckpt', 
                                    hparams_file='E:/projects/pixel-level-image-analysis/lightning_logs/version_0/hparams.yaml',
                                    map_location=torch.device('cpu'))
    
else:
    model = LBAE.load_from_checkpoint(checkpoint_path='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_0/checkpoints/epoch=99-step=2700.ckpt', 
                                    hparams_file='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_0/hparams.yaml',
                                    map_location=torch.device('cpu'))


test_dataset = myDataset(dataset_part='test')
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

X = torch.Tensor(test_dataset.dataset_data/16).reshape(len(test_dataset.dataset_data), 1, 8, 8)
preds = model(X)

X_test = X.clone().detach().reshape(360, 64)
preds = preds.clone().detach().reshape(360, 64)

print(f'Mean pairwise euclidean distance: {torch.mean(pairwise_euclidean_distance(X_test, preds))}')

def plot_images_from_tensors(tensor1, tensor2):
    # Sprawdź, czy dane są w tensorach PyTorch
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise ValueError("Both inputs should be PyTorch tensors.")

    # Wybierz pierwsze 5 obrazów z obu zbiorów
    images1 = tensor1[:9]
    images2 = tensor2[:9]

    # Tworzenie subplota
    fig, axes = plt.subplots(2, 9, figsize=(12, 5))

    # Pierwszy wiersz: Obrazy z pierwszego zbioru
    for i in range(len(images1)):
        image = images1[i].reshape(1, 8, 8)
        image = torch.permute(image, (1, 2, 0))
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title(f"X_true")
        axes[0, i].axis('off')

    # Drugi wiersz: Obrazy z drugiego zbioru
    for i in range(len(images2)):
        image = images2[i].reshape(1, 8, 8)
        image = torch.permute(image, (1, 2, 0))
        axes[1, i].imshow(image, cmap='gray')
        axes[1, i].set_title(f"Pred")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

plot_images_from_tensors(X, preds)