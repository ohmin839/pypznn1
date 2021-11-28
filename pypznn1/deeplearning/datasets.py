import os
import gzip
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypznn1.deeplearning.transforms import Compose, Flatten, ToFloat, Normalize
from pypznn1.deeplearning.utils import cache_dir, get_file

class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                    self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j

    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]

    return x, t

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)

class MNIST(Dataset):
    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                        'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                       'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

class CIFAR10(Dataset):
    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return
        filepath = get_file(url)
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty((50000), dtype=np.int)
            for i in range(5):
                self.data[i * 10000: (i + 1) * 10000] = self._load_data(filepath, i + 1, 'train')
                self.label[i * 10000: (i + 1) * 10000] = self._load_label(filepath, i + 1, 'train')
        else:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)

    def _load_data(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if (f'data_batch_{idx}' in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if (f'data_batch_{idx}' in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.array(data_dict[b'labels'])

    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H * row, W * col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

class ImageNet(Dataset):
    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels

class SinCurve(Dataset):
    def prepare(self):
        num_data = 1000
        dtype = np.float64
        
        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, +0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


def load_cache_npz(filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    suffix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + suffix)
    if not os.path.exists(filepath):
        return None, None
    
    loaded = np.load(filepath)
    return loaded['data'], loaded['label']

def save_cache_npz(data, label, filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    suffix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + suffix)

    if os.path.exists(filepath):
        return
    
    print("Saving: " + filename + suffix)
    try:
        np.savez_compressed(filepath, data=data, label=label)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print('Done')
    return filepath

