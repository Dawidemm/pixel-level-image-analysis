import numpy as np

np.random.seed(10)

bands = 220
channel = 1
w = 145
h = 145
total_samples = w * h

ground_truth = np.random.randint(0, 17, size=(channel, w, h))
hyperspectral = np.random.randint(0, 17, size=(bands, w, h))


def dataset_split(hyperspectral_image: np.array, ground_truth_image: np.array, split=0.2):

    gt_channel = ground_truth_image.shape[0]
    gt_w = ground_truth_image.shape[1]
    gt_h = ground_truth_image.shape[2]
    gt_total_samples = w * h

    h_bands = hyperspectral_image.shape[0]
    h_w = hyperspectral_image.shape[1]
    h_h = hyperspectral_image.shape[2]
    h_total_samples = h_w * h_h

    samples_to_remove_per_class = int(split * gt_total_samples)

    remaining_samples = []
    removed_samples = []

    ground_truth_data = ground_truth_image[0]
    ground_truth_data = ground_truth_data.flatten()
    indices_to_remove = np.random.choice(ground_truth_data.size, size=samples_to_remove_per_class, replace=False)
    remaining_indices = np.setdiff1d(np.arange(ground_truth_data.size), indices_to_remove)

    remaining_data = ground_truth_data[remaining_indices]
    removed_data = ground_truth_data[indices_to_remove]

    remaining_data = np.pad(remaining_data, (0, int(np.ceil(np.sqrt(len(remaining_data)))**2 - len(remaining_data))), mode='constant', constant_values=0)
    removed_data = np.pad(removed_data, (0, int(np.ceil(np.sqrt(len(removed_data)))**2 - len(removed_data))), mode='constant', constant_values=0)
    
    n1 = int(np.sqrt(len(remaining_data)))
    n2 = int(np.sqrt(len(removed_data)))
    remaining_data = remaining_data.reshape(n1, n1)
    removed_data = removed_data.reshape(n2, n2)

    remaining_samples.append(remaining_data)
    removed_samples.append(removed_data)

    remaining_samples = np.array(remaining_samples)
    removed_samples = np.array(removed_samples)

    hyperspectral_remaining_samples = []
    hyperspectral_removed_samples = []

    for band in range(h_bands):

        hyperspectral_data = hyperspectral_image[band]
        hyperspectral_data = hyperspectral_data.flatten()

        remaining_indices = np.setdiff1d(np.arange(hyperspectral_data.size), indices_to_remove)

        hyperspectral_remaining_data = hyperspectral_data[remaining_indices]
        hyperspectral_removed_data = hyperspectral_data[indices_to_remove]

        hyperspectral_remaining_data = np.pad(hyperspectral_remaining_data,
                                              (0, int(np.ceil(np.sqrt(len(hyperspectral_remaining_data)))**2 - len(hyperspectral_remaining_data))), 
                                              mode='constant', constant_values=0)
        
        hyperspectral_removed_data = np.pad(hyperspectral_removed_data,
                                            (0, int(np.ceil(np.sqrt(len(hyperspectral_removed_data)))**2 - len(hyperspectral_removed_data))),
                                            mode='constant', constant_values=0)
        
        n3 = int(np.sqrt(len(hyperspectral_remaining_data)))
        n4 = int(np.sqrt(len(hyperspectral_removed_data)))
        hyperspectral_remaining_data = hyperspectral_remaining_data.reshape(n3, n3)
        hyperspectral_removed_data = hyperspectral_removed_data.reshape(n4, n4)

        hyperspectral_remaining_samples.append(hyperspectral_remaining_data)
        hyperspectral_removed_samples.append(hyperspectral_removed_data)

    hyperspectral_remaining_samples = np.array(hyperspectral_remaining_samples)
    hyperspectral_removed_samples = np.array(hyperspectral_removed_samples)

    output = (hyperspectral_remaining_samples, hyperspectral_removed_samples, remaining_samples, removed_samples)

    return output


output = dataset_split(hyperspectral, ground_truth)

hyperspectral_remaining_samples = output[0]
hyperspectral_removed_samples = output[1]
remaining_samples = output[2]
removed_samples = output[3]

print('\n')
print('----------------------------------------------------------------------------------------')
print("Początkowe wymiary ground_truth", ground_truth.shape)
#print(ground_truth)
print('\n')
print("Wymiary obiektu z pozostałymi próbkami:", remaining_samples.shape)
#print(remaining_samples)
print('\n')
print("Wymiary obiektu z usuniętymi próbkami:", removed_samples.shape)
#print(removed_samples)

print('\n')
print('----------------------------------------------------------------------------------------')
print("Początkowe wymiary hyperspectral", hyperspectral.shape)
#print(hyperspectral)
print('\n')
print("Wymiary obiektu z pozostałymi próbkami:", hyperspectral_remaining_samples.shape)
#print(hyperspectral_remaining_samples)
print('\n')
print("Wymiary obiektu z usuniętymi próbkami:", hyperspectral_removed_samples.shape)

print('----------------------------------------------------------------------------------------')
#print(hyperspectral_removed_samples)