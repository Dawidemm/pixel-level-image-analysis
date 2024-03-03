import numpy as np

np.random.seed(10)


def train_test_split(hyperspectral_image: np.array, ground_truth_image: np.array, split=0.2):

    if ground_truth_image.shape[1] != hyperspectral_image.shape[1] or ground_truth_image.shape[2] != hyperspectral_image.shape[2]:
        raise ValueError('Dimension mismatch between ground truth image and hyperspectral image.')  

    total_samples = ground_truth_image.shape[1] * ground_truth_image.shape[2]
    samples_to_remove = int(split * total_samples)

    ground_truth_remaining_samples = []
    ground_truth_removed_samples = []

    ground_truth_image = ground_truth_image[0]
    ground_truth_image = ground_truth_image.flatten()
    indices_to_remove = np.random.choice(ground_truth_image.size, size=samples_to_remove, replace=False)
    remaining_indices = np.setdiff1d(np.arange(ground_truth_image.size), indices_to_remove)

    ground_truth_remaining_data = ground_truth_image[remaining_indices]
    ground_truth_removed_data = ground_truth_image[indices_to_remove]

    ground_truth_remaining_data = np.pad(ground_truth_remaining_data,
                            (0, int(np.ceil(np.sqrt(len(ground_truth_remaining_data)))**2 - len(ground_truth_remaining_data))),
                            mode='constant', 
                            constant_values=0)
    
    ground_truth_removed_data = np.pad(ground_truth_removed_data,
                          (0, int(np.ceil(np.sqrt(len(ground_truth_removed_data)))**2 - len(ground_truth_removed_data))),
                          mode='constant',
                          constant_values=0)
    
    ground_truth_remaining_data_shape = int(np.sqrt(len(ground_truth_remaining_data)))
    ground_truth_remaining_data = ground_truth_remaining_data.reshape(ground_truth_remaining_data_shape, ground_truth_remaining_data_shape)

    ground_truth_removed_data_shape = int(np.sqrt(len(ground_truth_removed_data)))
    ground_truth_removed_data = ground_truth_removed_data.reshape(ground_truth_removed_data_shape, ground_truth_removed_data_shape)

    ground_truth_remaining_samples.append(ground_truth_remaining_data)
    ground_truth_removed_samples.append(ground_truth_removed_data)

    ground_truth_remaining_samples = np.array(ground_truth_remaining_samples)
    ground_truth_removed_samples = np.array(ground_truth_removed_samples)

    hyperspectral_remaining_samples = []
    hyperspectral_removed_samples = []

    bands = hyperspectral_image.shape[0]

    for band in range(bands):

        hyperspectral_data = hyperspectral_image[band]
        hyperspectral_data = hyperspectral_data.flatten()

        remaining_indices = np.setdiff1d(np.arange(hyperspectral_data.size), indices_to_remove)

        hyperspectral_remaining_data = hyperspectral_data[remaining_indices]
        hyperspectral_removed_data = hyperspectral_data[indices_to_remove]

        hyperspectral_remaining_data = np.pad(hyperspectral_remaining_data,
                                              (0, int(np.ceil(np.sqrt(len(hyperspectral_remaining_data)))**2 - len(hyperspectral_remaining_data))), 
                                              mode='constant',
                                              constant_values=0)
        
        hyperspectral_removed_data = np.pad(hyperspectral_removed_data,
                                            (0, int(np.ceil(np.sqrt(len(hyperspectral_removed_data)))**2 - len(hyperspectral_removed_data))),
                                            mode='constant',
                                            constant_values=0)
        
        hyperspectral_remaining_data_shape = int(np.sqrt(len(hyperspectral_remaining_data)))
        hyperspectral_remaining_data = hyperspectral_remaining_data.reshape(hyperspectral_remaining_data_shape, hyperspectral_remaining_data_shape)

        hyperspectral_removed_data_shape = int(np.sqrt(len(hyperspectral_removed_data)))
        hyperspectral_removed_data = hyperspectral_removed_data.reshape(hyperspectral_removed_data_shape, hyperspectral_removed_data_shape)

        hyperspectral_remaining_samples.append(hyperspectral_remaining_data)
        hyperspectral_removed_samples.append(hyperspectral_removed_data)

    hyperspectral_remaining_samples = np.array(hyperspectral_remaining_samples)
    hyperspectral_removed_samples = np.array(hyperspectral_removed_samples)

    dataset = (hyperspectral_remaining_samples, 
              hyperspectral_removed_samples, 
              ground_truth_remaining_samples, 
              ground_truth_removed_samples)

    return dataset