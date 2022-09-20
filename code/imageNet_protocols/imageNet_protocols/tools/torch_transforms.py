import torchvision


"""

This code is copied from https://gitlab.ifi.uzh.ch/aiml/theses/master-bhoumik 
and has been wrapped into a function to embed it in the package.

The code is associated with the following Master's thesis:

  Bhoumik, Annesha (2021),
  Open-set Classification on ImageNet,
  Master's thesis,
  University of Zurich
  https://www.merlin.uzh.ch/publication/show/21577

"""


# image transformations for training data
def transform_train():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((300, 300)),
                                           torchvision.transforms.CenterCrop(
        (224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(
        10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])


# image transformations for validation and test data
def transform_val_test():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((300, 300)),
                                           torchvision.transforms.CenterCrop(
        (224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
