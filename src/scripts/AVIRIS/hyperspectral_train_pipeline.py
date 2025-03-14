# from torch.utils.data import DataLoader
# from src.utils.hyperspectral_dataset import AVIRISDataset
# import torch
# import numpy as np

# from src.qbm4eo.lbae import LBAE
# from src.qbm4eo.pipeline import Pipeline
# from src.qbm4eo.rbm import RBM

# torch.set_float32_matmul_precision('medium')

# np.random.seed(10)
# torch.manual_seed(0)

# NUM_VISIBLE = 55
# NUM_HIDDEN = 17

# MAX_EPOCHS = 25
# RBM_STEPS = 1000
# BATCH_SIZE = 16

# HYPERSPECTRAL_IMAGE_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'
# GROUND_TRUTH_IMAGE_PATH = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'

# def main():

#     try:
#         train_dataset = HyperspectralDataset(
#             hyperspectral_data=HYPERSPECTRAL_IMAGE_PATH,
#             ground_truth_data=GROUND_TRUTH_IMAGE_PATH,
#             stage=Stage.TRAIN
#         )

#     except FileNotFoundError as e:
#         print(f'FileNotFoundError: {e}')
#         print("Please make sure to provide paths to the hyperspectral image and ground truth image files.\n"
#               "The application will terminate now.")
#         return

#     train_dataloader = DataLoader(
#         dataset=train_dataset,
#         batch_size=BATCH_SIZE, 
#         shuffle=True, 
#         num_workers=4,
#         persistent_workers=True
#     )

#     autoencoder = LBAE(
#         input_size=(1, 220),
#         out_channels=8, 
#         latent_size=NUM_VISIBLE,
#         num_layers=2,
#         quantize=list(range(MAX_EPOCHS))
#     )
    
#     rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

#     pipeline = Pipeline(auto_encoder=autoencoder, rbm=rbm)

#     pipeline.fit(train_dataloader, max_epochs=MAX_EPOCHS, rbm_steps=RBM_STEPS, rbm_trainer='cd1', learnig_curve=True)

# if __name__ == '__main__':
#     main()


from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import AVIRISDataset
import torch
import numpy as np
import os

from src.qbm4eo.lbae import LBAE
from src.qbm4eo.pipeline import Pipeline
from src.qbm4eo.rbm import RBM

from torchmetrics.functional.pairwise import pairwise_euclidean_distance

torch.set_float32_matmul_precision('medium')

NUM_VISIBLE = 55
NUM_HIDDEN = 8

AUTOENCODER_EPOCHS = 20
AUTOENCODER_LEARNING_RATE = [0.001]
BATCH_SIZE = [8]

GROUND_TRUTH_DATA_PATH  = 'dataset/indian_pine/220x145x145/ground_truth_image.tif'
HYPERSPECTRAL_DATA_PATH = 'dataset/indian_pine/220x145x145/hyperspectral_image.tif'

RANDOM_SEED = 10

EXPERIMENT_FOLDER_PATH = './experiments/'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main():

    os.makedirs(EXPERIMENT_FOLDER_PATH, exist_ok=True)
    with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
        file.write(f'experiment,batch_size,learning_rate,pairwise_euclidean_distance,spectral_angle_distance\n')

    experiment = 0

    for learning_rate in AUTOENCODER_LEARNING_RATE:
        for batch_size in BATCH_SIZE:

            train_dataset = AVIRISDataset(
                hyperspectral_data='dataset/indian_pine/220x145x145/hyperspectral_image.tif',
                ground_truth_data='dataset/indian_pine/220x145x145/ground_truth_image.tif'
            )

            train_dataloader = DataLoader(
                dataset=train_dataset, 
                batch_size=batch_size
            )

            test_dataset = AVIRISDataset(
                hyperspectral_data='dataset/indian_pine/220x145x145/hyperspectral_image.tif',
                ground_truth_data='dataset/indian_pine/220x145x145/ground_truth_image.tif'
            )

            test_dataloader = DataLoader(
                dataset=test_dataset, 
                batch_size=1
            )

            lbae = LBAE(
                input_size=(1, 220),
                out_channels=8, 
                latent_size=NUM_VISIBLE,
                learning_rate=learning_rate,
                num_layers=2,
                quantize=list(range(AUTOENCODER_EPOCHS))
            )
                
            rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

            pipeline = Pipeline(auto_encoder=lbae, rbm=rbm)

            pipeline.fit(
                train_data_loader=train_dataloader, 
                validation_data_loader=None,
                autoencoder_epochs=AUTOENCODER_EPOCHS,
                skip_rbm=True,
                rbm_trainer='cd1',
                learnig_curve=True,
                experiment_folder_path=EXPERIMENT_FOLDER_PATH,
                experiment_number=experiment
            )

            mean_euclidean_distances = []

            with torch.no_grad():
                for X, _ in test_dataloader:
                    X_reconstructed = lbae(X)
                    X_reconstructed = X_reconstructed.reshape(X_reconstructed.shape[0]*X_reconstructed.shape[2], 1)
                    X = X.reshape(X.shape[0]*X.shape[2], 1)
                    distance = pairwise_euclidean_distance(X, X_reconstructed)
                    mean_euclidean_distances.append(torch.mean(distance))


            pairwise_euclidean_distance_mean = round(torch.mean(torch.tensor(mean_euclidean_distances)).item(), 3)

            with open(EXPERIMENT_FOLDER_PATH+'experiments_raport.csv', 'a+') as file:
                file.write(f'{experiment},{batch_size},{learning_rate},{pairwise_euclidean_distance_mean}\n')

            experiment += 1

if __name__ == '__main__':
    main()