import pdb
from torch.utils.data import Dataset,DataLoader
import pickle
from torchvision import datasets, transforms
class Mydataset(Dataset):
    def __init__(self, data, label):
        self.label = label
        self.data = data
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        x = transforms.ToTensor()(x)
        return x, y

def mnist_db(dataname,batch_size):
    with open("data/%s.pkl"%(dataname), 'rb') as f:
        x_train, y_train,x_test, y_test = pickle.load(f)
    traindataset = Mydataset(x_train, y_train)
    testdataset = Mydataset(x_test, y_test)
    train_dataloader =DataLoader(traindataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1) 
    valid_dataloader =DataLoader(testdataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=1) 
    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_loader, test_loader =mnist_db(batch_size=16)
    m1 = 'Length of iters per epoch: {}. Length of testing batches: {}.'
    print(m1.format(len(train_loader), len(test_loader)))
    data=next(iter(train_loader))
