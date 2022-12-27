import numpy as np
from .autograd import Tensor
import os
import pickle
import struct, gzip
import re
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img == True:
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        if shift_x <= 0:
            clip_x_up = shift_x + self.padding
            clip_x_down = clip_x_up + img.shape[0]
        else:
            clip_x_down = shift_x + self.padding + img.shape[0]
            clip_x_up = clip_x_down - img.shape[0]
        if shift_y <= 0:
            clip_y_up = shift_y + self.padding
            clip_y_down = clip_y_up + img.shape[1]
        else:
            clip_y_down = shift_y + self.padding + img.shape[1]
            clip_y_up = clip_y_down - img.shape[1]

        pad_img = np.pad(img, self.padding, 'constant')
        # print((clip_x_up, clip_x_down, clip_y_up, clip_y_down))
        return pad_img[clip_x_up : clip_x_down, clip_y_up : clip_y_down, self.padding:self.padding+img.shape[2]]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            shuffle_dataset = np.random.permutation(len(self.dataset))
            # print(shuffle_dataset)
            self.ordering = np.array_split(shuffle_dataset, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.num_batch = len(self.ordering)
        self.iterNum = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.iterNum < self.num_batch:
            # print(self.ordering[self.iterNum].tolist())
            batch_prec = self.dataset[self.ordering[self.iterNum].tolist()]
            if len(batch_prec) == 2:
                batch_X, batch_y = batch_prec
                batch = (Tensor(batch_X), Tensor(batch_y))
            elif len(batch_prec) == 1:
                batch_X, = batch_prec
                batch = (Tensor(batch_X), )
            self.iterNum = self.iterNum + 1
            return batch
        else:
            raise StopIteration 
        ### END YOUR SOLUTION

        # 我自己加的
    def __len__(self) -> int:
        return self.num_batch

def un_gz(filename):
  #解压缩.gz文件
  f_name = filename.replace(".gz", "")
  g_file = gzip.GzipFile(filename)
  open(f_name, "wb+").write(g_file.read())
  g_file.close()

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        path = '/content/drive/MyDrive/10714/hw4/'
        un_gz(path+label_filename)
        un_gz(path+image_filename)
        ungz_label_filename = label_filename.replace(".gz", "")
        ungz_image_filename = image_filename.replace(".gz", "")
        with open(path+ungz_label_filename, mode='rb') as labels, open(path+ungz_image_filename, mode='rb') as images:
            image_content = images.read(struct.calcsize('!4i'))
            _, image_num, image_height, image_width = struct.unpack("!4i", image_content)
            loaded = np.fromfile(file=images, dtype=np.uint8)

            max = loaded.max()
            min = loaded.min()
            X = loaded.reshape((image_num, image_height, image_width, 1)).astype(np.float32)
            #min-max normalization
            self.X = (X - min) / (max - min)
            
            label_content = labels.read(struct.calcsize('!2i'))
            _, label_num = struct.unpack("!2i", label_content)
            loaded = np.fromfile(file=labels, dtype=np.uint8)
            self.y = loaded.reshape((label_num,))

        self.transforms = transforms
        
        labels.close()
        images.close()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, slice):
            # print(index)
            img_slice = self.X[index.start:index.stop:index.step]
            y_slice = self.y[index.start:index.stop:index.step]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        elif isinstance(index, list):
            img_slice = self.X[index]
            y_slice = self.y[index]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        else:
            img = self.X[index]
            if self.transforms is not None:
                for transform in self.transforms:
                    img = transform(img)
            return (img, self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.p = p

        def load_cifar10_batch(batch_id=None):
            nonlocal base_folder, train
            if train:
                file_str = base_folder+'/data_batch_'+str(batch_id)
            else:
                file_str = base_folder+'/test_batch'
            with open(file_str, mode='rb') as file:
                batch = pickle.load(file, encoding='latin1')
            features, labels = (batch['data'] / 255).reshape((len(batch['data']), 3, 32, 32)), np.array(batch['labels'])
            return features, labels

        if train:
            self.X, self.y = load_cifar10_batch(batch_id=1)
            for i in range(2, 6):
                features, labels = load_cifar10_batch(batch_id=i)
                self.X, self.y = np.concatenate([self.X, features]), np.concatenate([self.y, labels])
        else:
            self.X, self.y = load_cifar10_batch()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index, slice):
            # print(index)
            img_slice = self.X[index.start:index.stop:index.step]
            y_slice = self.y[index.start:index.stop:index.step]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        elif isinstance(index, list):
            img_slice = self.X[index]
            y_slice = self.y[index]
            if self.transforms is not None and RandomCrop.random() < self.p:
                for transform in self.transforms:
                    img_slice = [transform(img) for img in img_slice]
            return (img_slice, y_slice)
        else:
            img = self.X[index]
            if self.transforms is not None and RandomCrop.random() < self.p:
                for transform in self.transforms:
                    img = transform(img)
            return (img, self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_count = {}

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx.keys():
            res = len(self.word2idx)
            self.word2idx[word] = res
            self.idx2word.append(word)
            self.word_count[word] = 1
        else:
            res = self.word2idx[word]
            self.word_count[word] += 1


        return res
        ### END YOUR SOLUTION
    def change_id_by_sort(self):
        """
        构造从单词到唯一整数值的映射
        后面的其他数的整数值按照它们在数据集里出现的次数多少来排序，出现较多的排前面
        单词 the 出现频次最多，对应整数值是 0
        <unk> 表示 unknown（未知），第二多，整数值为 1
        """
        word_count = sorted(self.word_count.items(), key=lambda x: -x[1])
        self.idx2word = [word[0] for word in word_count]
        self.word2idx = {self.idx2word[i]: i for i in range(len(self))}

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.word2idx)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        with open(path, 'r') as f:
            file_lines = f.readlines() if max_lines == None else f.readlines()[:max_lines]
            lines = [(line+" <eos>").strip().split() for line in file_lines]
            
            res = []

            # for line in lines:
            #     for word in line:
            #         self.dictionary.add_word(word)

            # self.dictionary.change_id_by_sort()

            for line in lines:
                for word in line:
                    res.append(self.dictionary.add_word(word))

            # print(f"'the' idx: {self.dictionary.word2idx['the']}, '<unk>' idx: {self.dictionary.word2idx['<unk>']}")

            return res
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    row = len(data) // batch_size
    return np.array(data[:row*batch_size], dtype=dtype).reshape(batch_size, row).transpose((1, 0))
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION

    seq_len = min(bptt, batches.shape[0] - 1 - i)
    data_org, target_org = nd.NDArray(batches[i:i+seq_len, :], device=device), nd.NDArray(batches[i+1:i+seq_len+1, :].reshape(-1), device=device)
    return Tensor(data_org, device=device, dtype=dtype), Tensor(target_org, device=device, dtype=dtype)
    ### END YOUR SOLUTION