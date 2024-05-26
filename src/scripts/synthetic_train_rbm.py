from torch.utils.data import DataLoader
from src.utils.hyperspectral_dataset import HyperspectralDataset, Stage

from src.qbm4eo.rbm import RBM, CD1Trainer
from src.utils import utils

NUM_VISIBLE = 220
NUM_HIDDEN = 17

RBM_STEPS = 1000
RBM_LEARNING_RATE = 0.001

SYNTH_IMG_SHAPE = 145
BATCH_SIZE = 16

def generator_dataloader(data_loader):
    while True:
        for batch_idx, (data, target) in enumerate(data_loader):
            yield data, target

def main():

    synthetic_data = utils.SyntheticDataGenerator(
        n_pixels=int(SYNTH_IMG_SHAPE*SYNTH_IMG_SHAPE),
        n_features=220,
        n_classes=17,
        image_width=SYNTH_IMG_SHAPE,
        image_height=SYNTH_IMG_SHAPE
    )   
    
    synthetic_hyperspectral_image, synthetic_ground_truth_image = synthetic_data.generate_synthetic_data(quantize=True)

    synthetic_train_dataset = HyperspectralDataset(
        hyperspectral_data=synthetic_hyperspectral_image,
        ground_truth_data=synthetic_ground_truth_image,
        stage=Stage.TRAIN
    )

    synthetic_train_dataloader = DataLoader(
        dataset=synthetic_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)

    print('RBM training with CD1Trainer.')
    rbm_trainer = CD1Trainer(RBM_STEPS, RBM_LEARNING_RATE)
    rbm_trainer.fit(rbm, generator_dataloader(synthetic_train_dataloader))
    rbm.save('rbm.npz')

    utils.plot_loss(
        epochs=rbm_trainer.num_steps, 
        loss_values=rbm_trainer.losses, 
        plot_title='RBM'
    )

    rbm = RBM(NUM_VISIBLE, NUM_HIDDEN)
    rbm.load(file='rbm.npz')


if __name__ == '__main__':
    main()