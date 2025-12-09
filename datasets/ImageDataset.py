from typing import Tuple
import random
import csv
import copy
import pandas as pd
import glob
import os
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image

from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations

class ImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """
  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int, target: str, train: bool, live_loading: bool, task: str) -> None:
    super(ImageDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task

    self.data = torch.load(data)
    self.labels = torch.load(labels)

    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
    self.transform_val = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])


  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    im = self.data[indx]
    if self.live_loading:
      im = read_image(im)
      im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.transform_val(im)
    
    label = self.labels[indx]
    return (im), label

  def __len__(self) -> int:
    return len(self.labels)

class CelebaImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """

  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int,
               target: str, train: bool, live_loading: bool, task: str) -> None:
    super(CelebaImageDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task

    self.data = self.read_image_files(data)
    self.tabular = self.read_and_parse_csv(labels)
    self.labels = self.tabular['Attractive']
    self.id = self.tabular['img_index']

    if delete_segmentation:
      pass

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
    self.transform_val = transforms.Compose([
      transforms.Resize(size=(img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

  def read_image_files(self, img_path) -> dict:
    """
    :return: {PetID: corresponding_image_name}
    """
    image_files = list(glob.glob(os.path.join(img_path, "*.jpg")))
    image_id_dict: Dict[str, List[str]] = defaultdict(list)
    for image_file in image_files:
      image_id = os.path.basename(image_file)
      image_id_dict[image_id].append(image_file)
    return image_id_dict

  def read_and_parse_csv(self, path_tabular: str) -> pd.DataFrame:
    """
    """
    data = pd.read_csv(path_tabular)
    return data

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    pet_id = self.id.iloc[index]
    rand_img_idx = random.randint(1, len(self.data[pet_id])) - 1
    im = self.data[pet_id][rand_img_idx]

    if self.live_loading:
      im = Image.open(im).convert('RGB')

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.transform_val(im)

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
    return (im), label

  def __len__(self) -> int:
    return len(self.tabular)

class SunImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """

  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int,
               target: str, train: bool, live_loading: bool, task: str) -> None:
    super(SunImageDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task

    self.data = self.read_image_files(data)
    self.tabular = self.read_and_parse_csv(labels)
    self.labels = self.tabular['open area']
    self.id = self.tabular['image_path']

    if delete_segmentation:
      pass

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
    self.transform_val = transforms.Compose([
      transforms.Resize(size=(img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

  def read_image_files(self, img_path) -> dict:
    """
    :return: {PetID: corresponding_image_name}
    """
    image_files = list(glob.glob(os.path.join(img_path, '**', '*.jpg'), recursive=True))
    image_id_dict: Dict[str, List[str]] = defaultdict(list)
    for image_file in image_files:
      image_id = os.path.relpath(image_file, img_path)
      image_id_dict[image_id].append(image_file)
    return image_id_dict

  def read_and_parse_csv(self, path_tabular: str) -> pd.DataFrame:
    """
    """
    data = pd.read_csv(path_tabular)

    return data

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    pet_id = self.id.iloc[index]
    rand_img_idx = random.randint(1, len(self.data[pet_id])) - 1
    im = self.data[pet_id][rand_img_idx]

    if self.live_loading:
      im = Image.open(im).convert('RGB')

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.transform_val(im)

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
    return (im), label

  def __len__(self) -> int:
    return len(self.tabular)