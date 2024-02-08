import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tifffile

HYPERSPECTRAL_IMAGE_PATH = 'dataset/hyperspectral_image.tif'
GROUND_TRUTH_IMAGE_PATH = 'dataset/ground_truth_image.tif'

def onehot(n):
    t = torch.zeros(17)
    t[n] = 1.0
    return t

class HyperspectralDataset(Dataset):
    def __init__(
            self, 
            hyperspectral_image_path: str, 
            ground_truth_image_path: str,
            stage: str
    ):
        hyperspectral_image = tifffile.imread(hyperspectral_image_path)
        ground_truth_image = tifffile.imread(ground_truth_image_path)

        hyperspectral_image = hyperspectral_image.astype(np.float32)
        ground_truth_image = ground_truth_image.astype(np.float32)

        hyperspectral_image = torch.tensor(hyperspectral_image)
        ground_truth_image = torch.tensor(ground_truth_image)
        
        hyperspectral_image /= hyperspectral_image.max()
        ground_truth_image = ground_truth_image.reshape(1, ground_truth_image.shape[0], ground_truth_image.shape[1])

        self.hyperspectral_image = hyperspectral_image
        self.ground_truth_image = ground_truth_image

        # if stage == 'train':
        #     pass
        # elif stage == 'test':
        #     pass
        # else:
        #     raise ValueError(f'Stage should be set as ["train", "test"].')

        self.image_shape = hyperspectral_image.shape
        self.bands = self.image_shape[0]
        self.width = self.image_shape[1]
        self.height = self.image_shape[2]

    def  __len__(self):
        return self.width * self.height
    
    def __getitem__(self, index):

        x = index // self.width
        y = index % self.height

        pixel_values = self.hyperspectral_image[:, x, y]
        padding = torch.zeros(36)
        combined_pixel_values = torch.cat((pixel_values, padding), dim=0)
        combined_pixel_values = combined_pixel_values.reshape(1, 16, 16)

        label = self.ground_truth_image.clone().detach()
        label = int(label[:, x, y].item())
        label = onehot(label)
        
        return combined_pixel_values, label
    
# data = HyperspectralDataset(
#     hyperspectral_image_path='dataset/hyperspectral_image.tif',
#     ground_truth_image_path='dataset/ground_truth_image.tif'
# )
# print(data.bands)

# dataloader = DataLoader(dataset=data, batch_size=8, shuffle=False)

# combined_tensor = torch.empty(0)

# for batch, (X, y) in enumerate(dataloader):
#     print(f'batch: {batch}')
#     print(f'X: {X.shape}')
#     print(f'X: {X}')
#     print(f'y: {y.shape}')
#     print(f'y: {y}')

#     combined_tensor = torch.cat((combined_tensor, y), dim=0)

# combined_tensor = combined_tensor.reshape(1, 145, 145)
ground_truth_image = tifffile.imread('dataset/ground_truth_image.tif')
ground_truth_image = ground_truth_image.astype(np.float32)
unique_classes, counts = np.unique(ground_truth_image, return_counts=True)

for i in range(len(unique_classes)):
    print('---------------------------------------------')
    print("Unikalne wartości w macierzy:", unique_classes[i])
    print(f'Liczba unikalnych wartości: {counts[i]}. 20%: {int(counts[i] * 0.2)}')
    print('\n')

# Obliczamy liczbę unikalnych klas
num_classes = len(np.unique(ground_truth_image))

# Obliczamy liczbę pikseli do usunięcia z każdej klasy (20% każdej klasy)
pixels_to_remove_per_class = {cls: int(np.sum(ground_truth_image == cls) * 0.2) for cls in range(num_classes)}
print(pixels_to_remove_per_class)

# Kopiujemy oryginalną mapę klasyfikacji
new_ground_truth_image = ground_truth_image.copy()

# Tworzymy tensor, który będzie zawierał tylko usunięte wartości
removed_values_tensor = np.zeros((0, 3), dtype=np.int32)  # Początkowo pusty tensor

# Usuwamy 20% pikseli z każdej klasy
for cls in range(num_classes):
    class_pixels = np.where(ground_truth_image == cls)  # Indeksy pikseli dla danej klasy
    pixels_to_remove = np.random.choice(range(len(class_pixels[0])), size=pixels_to_remove_per_class[cls], replace=False)
    print(f'class: {cls}')
    print(f'pixel to remove indexes: {pixels_to_remove.shape} \n{pixels_to_remove}')
    
    # Usuwamy wybrane piksele z klasy
    new_ground_truth_image[class_pixels[0][pixels_to_remove], class_pixels[1][pixels_to_remove]] = 0
    
    # Dodajemy usunięte wartości do tensora
    removed_values = np.vstack((class_pixels[0][pixels_to_remove], class_pixels[1][pixels_to_remove], np.full(len(pixels_to_remove), cls))).T
    removed_values_tensor = np.vstack((removed_values_tensor, removed_values))

print("Mapa klasyfikacji z usuniętymi pikselami:", new_ground_truth_image.shape)
print("Tensor zawierający usunięte wartości:", removed_values_tensor.shape)

# ground_truth_image = torch.tensor(ground_truth_image)

# result = torch.allclose(combined_tensor, ground_truth_image)
# print(result)