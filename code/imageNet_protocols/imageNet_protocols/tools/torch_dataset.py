# import libraries
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd
from . import torch_transforms


"""

This code is copied from https://gitlab.ifi.uzh.ch/aiml/theses/master-bhoumik.
An additional 'BG' parameter has been added to embed the TorchDataset class into the package.
Further, the following attributes have been added:
  self.csv_file             # attribute; path to csv file
  self.num_kk_classes       # attribute; number of known known classes



The code is associated with the following Master's thesis:

  Bhoumik, Annesha (2021),
  Open-set Classification on ImageNet,
  Master's thesis,
  University of Zurich
  https://www.merlin.uzh.ch/publication/show/21577

"""


class TorchDataset (Dataset):

    def __init__(self, csv_file, images_path, transform=None, BG=False):
        """
            :param csv_file: String, path to the csv file with image paths.
            :param images_path: String, directory with all the images.
            :param transform: Callable, optional transform to be applied on a sample.
            :param BG: Boolean, use of a background class in the model for known unknowns
        """
        # read train/val/test csv file
        self.dataset = pd.read_csv(csv_file, header=None)
        self.csv_source = csv_file
        self.num_kk_classes = self.dataset[1].max() + 1
        # if method = Softmax with Garbage Class, then replace -1 label for known unknown classes
        # with (Number of known classes)
        if BG:
            self.dataset[1].replace(-1,
                                    (self.dataset[1].max()) + 1, inplace=True)
        self.images_path = images_path
        self.transform = transform

    # return size of dataset
    def __len__(self):
        return len(self.dataset)

    # return sample from the dataset given an index
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # path to image within a class from csv file
        jpeg_path = self.dataset.iloc[index, 0]
        # corresponding integer label for image
        label = self.dataset.iloc[index, 1]
        # use RGB mode to represent and store the image
        img = Image.open(self.images_path + jpeg_path).convert('RGB')
        # apply image transformation
        if self.transform is not None:
            x = self.transform(img)
        # convert integer label to tensor
        t = torch.as_tensor(int(label), dtype=torch.int64)
        # return image, label
        return x, t


# create custom train, validation and test datasets
def generate_datasets(data_path, train_file_csv, val_file_csv, test_file_csv, BG=False, print_info=False):
    """
    Args:
        data_path (string, pathlib.Path): filesystem path to root directory of the ILSVRC2012 dataset
        train_file_csv (string, pathlib.Path): filesystem path to protocol-generated csv file used for creating training dataset
        val_file_csv (string, pathlib.Path): filesystem path to protocol-generated csv file used for creating validation dataset
        test_file_csv (string, pathlib.Path): filesystem path to protocol-generated csv file used for creating test dataset
        print_info (bool): indicates whether to print information of the generated datasets onto the standard output. Note that the function returns these info messages anyway.
    """
    train_set, val_set, test_set = None, None, None
    msg_train, msg_val, msg_test = None, None, None

    if train_file_csv is not None:
        train_set = TorchDataset(
            csv_file=train_file_csv, images_path=data_path, transform=torch_transforms.transform_train(), BG=BG)
        msg_train = f'Data source training set: {train_set.csv_source}\t# of kkc: {train_set.num_kk_classes}\tSamples: {train_set.__len__()}'
        if print_info:
            print(f'{msg_train}')

    if val_file_csv is not None:
        val_set = TorchDataset(
            csv_file=val_file_csv, images_path=data_path, transform=torch_transforms.transform_val_test(), BG=BG)
        msg_val = f'Data source validation set: {val_set.csv_source}\t# of kkc: {val_set.num_kk_classes}\tSamples: {val_set.__len__()}'
        if print_info:
            print(f'{msg_val}')

    if test_file_csv is not None:
        test_set = TorchDataset(
            csv_file=test_file_csv, images_path=data_path, transform=torch_transforms.transform_val_test(), BG=BG)
        msg_test = f'Data source test set: {test_set.csv_source}\t# of kkc: {test_set.num_kk_classes}\tSamples: {test_set.__len__()}'
        if print_info:
            print(f'{msg_test}')

    return train_set, val_set, test_set, [msg_train, msg_val, msg_test]
