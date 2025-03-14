import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score, adjusted_rand_score, rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans

from src.utils.blood_dataset import BloodIterableDataset, Stage
from src.utils import utils
from src.qbm4eo.lbae import LBAE
from src.qbm4eo.rbm import RBM

np.random.seed(10)
torch.manual_seed(0)

NUM_VISIBLE = 28
NUM_HIDDEN = 23

HYPERSPECTRAL_DATA_PATH = 'HyperBlood/data'
GROUND_TRUTH_DATA_PATH = 'HyperBlood/anno'
AUTOENCODER_CHECKPOINT_PATH = 'lightning_logs/version_3/checkpoints/epoch=19-step=40000.ckpt'
AUTOENCODER_HPARAMS_PATH = 'lightning_logs/version_3/hparams.yaml'


# MODEL = 'experiments_SA/exp_5/rbm_nh=23_seed=50_epoch=200.npz'
MODEL = 'experiments_QA/exp_1/rbm_nh=23_seed=10_epoch=300.npz'

IMAGES = ['E_7']

LINKAGE = ['single', 'complete', 'average']

THRESHOLDS = np.linspace(1/25, 1, 25)[:-1]


def main():

    rbm = RBM(num_visible=NUM_VISIBLE, num_hidden=NUM_HIDDEN, random_seed=70)
    rbm = rbm.load(file=MODEL)

    seg_dataset = BloodIterableDataset(
        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
        load_specific_images=IMAGES,
        stage=Stage.SEG,
        remove_noisy_bands=True,
        remove_background=True,
        shuffle=False
    )

    seg_dataloader = DataLoader(dataset=seg_dataset, batch_size=256, drop_last=False)

    lbae = LBAE.load_from_checkpoint(
        checkpoint_path=AUTOENCODER_CHECKPOINT_PATH, 
        hparams_file=AUTOENCODER_HPARAMS_PATH,
        map_location=torch.device('cpu')
    )
    lbae.eval()

    threshold_finder = utils.ThresholdFinder(
                dataloader=seg_dataloader,
                rbm=rbm,
                encoder=lbae.encoder
            )

    threshold, rbm_ars, rbm_rand_score, rbm_homogeneity, rbm_completeness, num_unique_labels = threshold_finder.find_threshold(THRESHOLDS)

    hidden_representations = []
    y_true = []
    X_true = []
    rbm_labels = []

    seg_dataloader = DataLoader(dataset=seg_dataset, batch_size=1, drop_last=False)
    
    with torch.no_grad():
        for idx, (X, y) in enumerate(seg_dataloader):
            hidden_representation, _ = lbae.encoder(X, epoch=1)
            hidden_representation = hidden_representation.detach().numpy()
            hidden_representations.append(hidden_representation)

            rbm_label = rbm.binarized_rbm_output(hidden_representation, threshold=threshold)
            rbm_labels.append(rbm_label)

            y_true.append(y)
            X_true.append(X)

    hidden_representations = np.concatenate(hidden_representations)
    y_true = np.concatenate(y_true)
    X_true = np.concatenate(X_true)
    X_true = np.reshape(X_true, (X_true.shape[0], X_true.shape[2]))
    rbm_labels = np.concatenate(rbm_labels)
    
    # data = np.load('distance_matrices/RBM_nh23_rs70_E_7_dist_matrix.npz')
    # distance_matrix = data['arr_0']

    # distance_matrix = utils.compute_distance_matrix(hidden_representations, rbm_labels)

    kmeas = KMeans(n_clusters=6)
    labels = kmeas.fit_predict(X_true)

    # for link in LINKAGE:

    #     ahc = AgglomerativeClustering(n_clusters=6, metric='precomputed', linkage=link)
    #     print('Start agglomerative clustering...')
    #     labels = ahc.fit_predict(distance_matrix)
    #     labels = labels+1

    #     ahc_homogenity = homogeneity_score(y_true, labels)
    #     ahc_completeness = completeness_score(y_true, labels)
    #     ahc_v_measure = v_measure_score(y_true, labels)
    #     ahc_ari = adjusted_rand_score(y_true, labels)
    #     ahc_rand_score = rand_score(y_true, labels)

    #     with open('metrics_'+link+'.txt', 'w') as f:
    #         f.write(f"After RBM pre-clustering:\n")
    #         f.write(f"Homogeneity: {rbm_homogeneity}\n")
    #         f.write(f"Completeness: {rbm_completeness}\n")
    #         f.write(f"ARI: {rbm_ars}\n")
    #         f.write(f"Rand Score: {rbm_rand_score}\n")

    #         f.write(f"After AHC clustering:\n")
    #         f.write(f"Homogeneity: {ahc_homogenity}\n")
    #         f.write(f"Completeness: {ahc_completeness}\n")
    #         f.write(f"ARI: {ahc_ari}\n")
    #         f.write(f"Rand Score: {ahc_rand_score}")

    ahc_homogenity = homogeneity_score(y_true, labels)
    ahc_completeness = completeness_score(y_true, labels)
    ahc_v_measure = v_measure_score(y_true, labels)
    ahc_ari = adjusted_rand_score(y_true, labels)
    ahc_rand_score = rand_score(y_true, labels)

    # with open('enc_data_metrics_kmease_seg.txt', 'w') as f:
    #     f.write(f"After RBM pre-clustering:\n")
    #     f.write(f"Homogeneity: {rbm_homogeneity}\n")
    #     f.write(f"Completeness: {rbm_completeness}\n")
    #     f.write(f"ARI: {rbm_ars}\n")
    #     f.write(f"Rand Score: {rbm_rand_score}\n")

    #     f.write(f"Kmeans clustering:\n")
    #     f.write(f"Homogeneity: {ahc_homogenity}\n")
    #     f.write(f"Completeness: {ahc_completeness}\n")
    #     f.write(f"ARI: {ahc_ari}\n")
    #     f.write(f"Rand Score: {ahc_rand_score}")

    img_dataset = BloodIterableDataset(
        hyperspectral_data_path=HYPERSPECTRAL_DATA_PATH,
        ground_truth_data_path=GROUND_TRUTH_DATA_PATH,
        load_specific_images=IMAGES,
        stage=Stage.SEG,
        remove_noisy_bands=True,
        remove_background=False,
        shuffle=False
    )

    img_dataloader = DataLoader(dataset=img_dataset, batch_size=1, drop_last=False)

    segmented_img = []
    label_index = 0

    for idx, (X, y) in enumerate(img_dataloader):
        if y.item() == 0:
            segmented_img.append(0)
        else:
            segmented_img.append(labels[label_index])
            label_index += 1

    segmented_img = np.array(segmented_img)
    segmented_img = np.reshape(segmented_img, (520, 696))

    plt.figure(figsize=(4, 3))
    plt.imshow(segmented_img)
    plt.axis('off')
    plt.tight_layout()
    # plt.savefig('rbm_nh=23_seed=10_epoch=300_link='+link+'.pdf', format='pdf', dpi=300)
    plt.savefig('raw_kmeans.pdf', format='pdf', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
