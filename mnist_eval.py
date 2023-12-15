import torch
import lightning as pl
import numpy as np
from myDataset import myDataset
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from torchmetrics.clustering import RandScore
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score

from lbae import LBAE
from rbm import RBM

if torch.cuda.is_available():
    lbae = LBAE.load_from_checkpoint(checkpoint_path='E:/projects/pixel-level-image-analysis/lightning_logs/version_24/checkpoints/epoch=9-step=1800.ckpt', 
                                    hparams_file='E:/projects/pixel-level-image-analysis/lightning_logs/version_24/hparams.yaml',
                                    map_location=torch.device('cpu'))
    
else:
    lbae = LBAE.load_from_checkpoint(checkpoint_path='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_21/checkpoints/epoch=49-step=9000.ckpt', 
                                    hparams_file='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_21/hparams.yaml',
                                    map_location=torch.device('cpu'))


test_dataset = myDataset(dataset_part='test')
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

X = torch.Tensor(test_dataset.dataset_data/16).reshape(len(test_dataset.dataset_data), 1, 8, 8)

preds = lbae(X)

X_test = X.clone().detach().reshape(360 * 64, 1)
preds = preds.clone().detach().reshape(360 * 64, 1)

print(f'Mean pairwise euclidean distance: {torch.mean(pairwise_euclidean_distance(X_test, preds))}')

X_test = X.clone().detach().reshape(360, 64)
preds = preds.clone().detach().reshape(360, 64)

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

# plot_images_from_tensors(X, preds)

def bin_rbm_out(h_probs_given_v: np.array, threshold: float) -> np.array:

    h_probs_given_v[h_probs_given_v <= threshold] = 0
    h_probs_given_v[h_probs_given_v > threshold] = 1

    return h_probs_given_v

def map_to_indices(values, lst):
    indices = [lst.index(value) for value in values]
    return indices


NUM_VISIBLE = 60
NUM_HIDDEN = 40
MAX_EPOCHS = 100

rbm_model = RBM(NUM_VISIBLE, NUM_HIDDEN)
rbm_model.load(file='rbm.npz')

def try_thresholds(thresholds: int, data: torch.Tensor, y_true: torch.Tensor, with_best=False):

    if with_best == False:
        thresholds = np.linspace(1/thresholds, thresholds/thresholds, thresholds)
    else:
        thresholds = [0.73]

    rand_scores = []
    mapped_probs_l = []

    for threshold in thresholds:

        unique_probs = set()
        probs = []

        for i in range(360):
            
            Xi = data[i].reshape(1, 1, 8, 8)
            encoder, err = lbae.encoder.forward(Xi, epoch=1)

            rbm_input = encoder.detach().numpy()
            prob = rbm_model.h_probs_given_v(rbm_input)
            prob = bin_rbm_out(prob, threshold)
            unique_prob = tuple(map(tuple, prob))
            unique_probs.add(unique_prob)

            prob = tuple(map(tuple, prob))
            probs.append(prob)

        unique_probs = list(unique_probs)

        mapped_probs = map_to_indices(probs, unique_probs)

        mapped_probs = np.array(mapped_probs)
        mapped_probs_l.append(mapped_probs)

        y_true = np.array(y_true)

        rs = rand_score(y_true, mapped_probs)
        rand_scores.append(rs)

        print(f'\nth: {round(threshold, 3)}, rand score: {rs}')

    rand_scores = np.array(rand_scores)
    rand_score_max_index = np.argmax(rand_scores)
    best_threshold = thresholds[rand_score_max_index]
    best_rand_score = np.max(rand_scores)

    return best_threshold, best_rand_score

y_true = torch.Tensor(test_dataset.dataset_labels)

th, rs = try_thresholds(thresholds=100, data=X, y_true=y_true, with_best=True)

print(f'\nBest threshold: {th}\nBest Rand score: {rs}\n')