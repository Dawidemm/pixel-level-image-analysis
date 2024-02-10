from sklearn import datasets
import torch
from torch.utils.data import DataLoader, Dataset

def onehot(n):
    t = torch.zeros(10)
    t[n] = 1.0
    return t

class myDataset(Dataset):
    def __init__(self, dataset_part: str):

        digits = datasets.load_digits()

        self.dataset_data = digits.data
        self.dataset_labels = digits.target

        self.dataset_part = dataset_part

        if self.dataset_part == 'train':
            self.dataset_data = self.dataset_data[:int(0.8 * len(self.dataset_data))]
            self.dataset_labels = self.dataset_labels[:int(0.8 * len(self.dataset_labels))]
            
        elif self.dataset_part == 'test':
            self.dataset_data = self.dataset_data[int(0.8 * len(self.dataset_data)):]
            self.dataset_labels = self.dataset_labels[int(0.8 * len(self.dataset_labels)):]
        
        else:
            raise ValueError('Pick one from [train, test] datasets')

    def __getitem__(self, index) -> tuple:

        digit = torch.Tensor(self.dataset_data[index])
        digit = digit/torch.max(torch.Tensor(self.dataset_data))
        digit = torch.reshape(digit, (1, 8, 8))

        label = list(map(onehot, list(self.dataset_labels)))
        label = torch.Tensor(label[index])

        return digit, label
    
    def  __len__(self) -> int:
        return self.dataset_labels.shape[0]
    
def main():
    train_dataset = myDataset(dataset_part='train')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=4)

    num_ep = 1

    for epoch in range(num_ep):
        for batch_index, (x, y) in enumerate(train_dataloader):
            if batch_index >= 3:
                break

            print(f'Batch index: {batch_index}')
            print(f'Batch size: {y.shape[0]}')
            print(f'x shape: {x.shape}')
            print(f'x first image: {x[0][0]}')
            print(f'y shape: {y.shape}')
            print(f'y: {y}')

if __name__ == '__main__':
    main()