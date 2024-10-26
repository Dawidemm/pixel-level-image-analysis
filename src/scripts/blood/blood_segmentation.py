# import os
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score, adjusted_rand_score, rand_score

# from src.utils.blood_dataset import BloodIterableDataset, Stage

# from src.utils import utils

# from src.qbm4eo.lbae import LBAE
# from src.qbm4eo.rbm import RBM
# from src.qbm4eo.ahc import AgglomerativeHierarchicalClustering

# from src.utils import utils

# import matplotlib.pyplot as plt

# np.random.seed(10)
# torch.manual_seed(0)

# NUM_VISIBLE = 28
# NUM_HIDDEN = 23
# RANDOM_SEEDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


# HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
# GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'

# AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=19-step=290280.ckpt'
# AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'
# RBM_MODEL_PATH = 'experiments/exp_203/rbm.npz'

# IMAGES = ['F_1']

# def main():

#     test_labels = np.array([])

#     seg_dataset = BloodIterableDataset(
#         hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
#         ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
#         load_specific_images=IMAGES,
#         stage=Stage.SEG,
#         remove_noisy_bands=True,
#         remove_background=True,
#         shuffle=False
#     )

#     seg_dataloader = DataLoader(
#             dataset=seg_dataset,
#             batch_size=1,
#             drop_last=False
#         )

#     lbae = LBAE.load_from_checkpoint(
#         checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
#         hparams_file=AUTOENCODER_HPARAMS_PATH,
#         map_location=torch.device('cpu')
#     )

#     lbae.eval()

#     rbm = RBM(
#         num_visible=NUM_VISIBLE,
#         num_hidden=NUM_HIDDEN,
#         random_seed=RANDOM_SEED
#     )

#     rbm.load(file=RBM_MODEL_PATH)

#     y_true = []
#     hidden_representations = []
#     rbm_labels = []

#     print('Collectrring data')
#     with torch.no_grad():
#         for idx, (X, y) in enumerate(seg_dataloader):
#             if y == 0:
#                 continue
#             else:
#                 hidden_representation, _ = lbae.encoder(X, epoch=1)
#                 hidden_representation = hidden_representation.detach().numpy()
#                 rbm_label = rbm.binarized_rbm_output(hidden_representation, threshold=0.4)
            
#                 hidden_representations.append(hidden_representation)
#                 rbm_labels.append(rbm_label)

#                 y_true.append(y)

#     hidden_representations = np.concatenate(hidden_representations)
#     y_true = np.concatenate(y_true)

#     print('Computing RBM labels...')
#     th_finder = utils.ThresholdFinder(
#         dataloader=seg_dataloader,
#         rbm=rbm,
#         encoder=lbae.encoder
#     )
#     _, _, _, _, _, _, mapped_rbm_labels = th_finder.find_threshold(thresholds=[0.7])

#     print('Strat agglomerative clustering...')
#     ahc = AgglomerativeHierarchicalClustering(n_clusters=7, linkage="single")
#     labels = ahc.fit(X=hidden_representations, initial_labels=mapped_rbm_labels)

#     test_labels = np.append(test_labels, labels)

#     homogenity = homogeneity_score(y_true, test_labels)
#     completeness = completeness_score(y_true, test_labels)

#     print(f'Homogenity: {round(homogenity, 3)}')
#     print(f'Completeness: {round(completeness, 3)}')

#     seg_dataset = BloodIterableDataset(
#             hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
#             ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
#             load_specific_images=IMAGES,
#             stage=Stage.SEG,
#             remove_noisy_bands=True,
#             remove_background=False,
#             shuffle=False
#         )
    
#     seg_dataloader = DataLoader(
#                 dataset=seg_dataset,
#                 batch_size=1,
#                 drop_last=False
#             )

#     segmented_img = []
#     counter = 0

#     for idx, (X, y) in enumerate(seg_dataloader):
#         if y.item() == 0:
#             segmented_img.append(y.item())
#         else:
#             segmented_img.append(test_labels[counter])
#             counter += 1

#     segmented_img = np.array(segmented_img)
#     segmented_img = np.reshape(segmented_img, ((520, 696)))
#     np.save('ahc', segmented_img)

#     plt.figure(figsize=(4, 3))
#     plt.imshow(segmented_img)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('ahc.png', dpi=300)


# if __name__ == '__main__':
#     main()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score, adjusted_rand_score, rand_score

from src.utils.blood_dataset import BloodIterableDataset, Stage
from src.utils import utils
from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM
from src.qbm4eo.ahc import AgglomerativeHierarchicalClustering

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 28
NUM_HIDDEN = 23

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'
AUTOENCODER_CHECKPOINT_PATH = 'model/epoch=19-step=290280.ckpt'
AUTOENCODER_HPARAMS_PATH = 'model/hparams.yaml'
RBM_MODELS_DIR = 'model/rbms'
SEGMENTATION_OUTPUT_DIR = 'segmentation_results'

# THRESHOLDS = np.linspace(1/10, 1, 10)[:-1]
THRESHOLDS = [0.6, 0.5, 0.8, 0.2, 0.4, 0.8, 0.3, 0.8, 0.9, 0.4]

IMAGES = ['F_1']

def main():
    os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

    # Iteracja przez modele RBM
    for rbm_file in os.listdir(RBM_MODELS_DIR):
        for threshold in THRESHOLDS:
            rbm_path = os.path.join(RBM_MODELS_DIR, rbm_file)
            
            model_result_dir = os.path.join(SEGMENTATION_OUTPUT_DIR, os.path.splitext(rbm_file)[0])
            os.makedirs(model_result_dir, exist_ok=True)

            rbm = RBM(num_visible=NUM_VISIBLE, num_hidden=NUM_HIDDEN)
            rbm.load(file=rbm_path)

            seg_dataset = BloodIterableDataset(
                hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                load_specific_images=IMAGES,
                stage=Stage.SEG,
                remove_noisy_bands=True,
                remove_background=True,
                shuffle=False
            )

            seg_dataloader = DataLoader(dataset=seg_dataset, batch_size=1, drop_last=False)

            lbae = LBAE.load_from_checkpoint(
                checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
                hparams_file=AUTOENCODER_HPARAMS_PATH,
                map_location=torch.device('cpu')
            )
            lbae.eval()

            hidden_representations = []
            y_true = []
            
            with torch.no_grad():
                for idx, (X, y) in enumerate(seg_dataloader):
                    if y == 0:
                        continue
                    hidden_representation, _ = lbae.encoder(X, epoch=1)
                    hidden_representation = hidden_representation.detach().numpy()
                    hidden_representations.append(hidden_representation)
                    y_true.append(y)

            hidden_representations = np.concatenate(hidden_representations)
            y_true = np.concatenate(y_true)


            th_finder = utils.ThresholdFinder(dataloader=seg_dataloader, rbm=rbm, encoder=lbae.encoder)
            threshold, rbm_ari, rbm_rand_score, rbm_homogenity, rbm_completeness, rbm_v_measure_scores, rbm_mapped_labels = th_finder.find_threshold(thresholds=[threshold])

            ahc = AgglomerativeHierarchicalClustering(n_clusters=7, linkage="single")
            labels = ahc.fit(X=hidden_representations, initial_labels=rbm_mapped_labels)

            ahc_homogenity = homogeneity_score(y_true, labels)
            ahc_completeness = completeness_score(y_true, labels)
            ahc_v_measure = v_measure_score(y_true, labels)
            ahc_ari = adjusted_rand_score(y_true, labels)
            ahc_rand_score = rand_score(y_true, labels)
            
            metrics_path = os.path.join(model_result_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"After RBM pre-clustering:")
                f.write(f"RBM's homogeneity: {rbm_homogenity}\n")
                f.write(f"RBM's completeness: {rbm_completeness}\n")
                f.write(f"RBM's v-measure: {np.mean(rbm_v_measure_scores)}\n")
                f.write(f"RBM's ari: {rbm_ari}\n")
                f.write(f"RBM's rand score: {rbm_rand_score}")

                f.write(f"After AHC clustering:")
                f.write(f"RBM's homogeneity: {ahc_homogenity}\n")
                f.write(f"RBM's completeness: {ahc_completeness}\n")
                f.write(f"RBM's v-measure: {ahc_v_measure}\n")
                f.write(f"RBM's ari: {ahc_ari}\n")
                f.write(f"RBM's rand score: {ahc_rand_score}")

            seg_dataset = BloodIterableDataset(
                hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
                ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
                load_specific_images=IMAGES,
                stage=Stage.SEG,
                remove_noisy_bands=True,
                remove_background=False,
                shuffle=False
            )

            seg_dataloader = DataLoader(dataset=seg_dataset, batch_size=1, drop_last=False)

            segmented_img = []
            counter = 0
            for idx, (X, y) in enumerate(seg_dataloader):
                if y.item() == 0:
                    segmented_img.append(y.item())
                else:
                    segmented_img.append(labels[counter])
                    counter += 1

            segmented_img = np.array(segmented_img)
            segmented_img = np.reshape(segmented_img, (520, 696))

            img_path = os.path.join(model_result_dir, 'segmentation.pdf')
            plt.figure(figsize=(4, 3))
            plt.imshow(segmented_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img_path, format='pdf', dpi=300)
            plt.close()

            print(f'Finished processing {rbm_file}. Results saved in {model_result_dir}.')

if __name__ == '__main__':
    main()
