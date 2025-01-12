import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision

# Directory containing the data.
root = 'data/'

class MovingMNISTDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.batch_size = dataloader.batch_size
        self.num_frames = 20

    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        return self

    def __next__(self):
        batch = next(self.dataloader_iter)
        
        frames = batch.view(self.batch_size * self.num_frames, 1, 64, 64).float()/255.0
        # downsample to 32x32
        frames = torch.nn.functional.interpolate(frames, size=(28, 28), mode='bilinear')
        
        return frames

    def __len__(self):
        return len(self.dataloader)

def get_data(dataset, batch_size):
    dataset_type = dataset
    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/', train='train', 
                                download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root+'svhn/', split='train', 
                                download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)
    elif dataset == 'MovingMNIST':
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(32),
        #     transforms.ToTensor(),
        #     ])
        transform = None
        
        dataset = torchvision.datasets.MovingMNIST(root = root+'movingmnist/', download=True, transform=transform)[:].view(-1, 1, 64, 64).float()/255
        # downsample to 32x32
        dataset = torch.nn.functional.interpolate(dataset, size=(28, 28), mode='bilinear')
        # batch_size = batch_size // 20 # 20 frames per video
    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=16)
    
    # if dataset_type == 'MovingMNIST':
    #     dataloader = MovingMNISTDataLoader(dataloader)

    return dataloader