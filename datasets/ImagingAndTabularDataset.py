from typing import List, Tuple
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
import numpy as np

from utils.utils import grab_hard_eval_image_augmentations

class ImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for evaluation.

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view is never augmented.
  """
  def __init__(
      self,
      data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float, 
      data_path_tabular: str, field_lengths_tabular: str, eval_one_hot: bool,
      labels_path: str, img_size: int, live_loading: bool, train: bool, target: str) -> None:
      
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.eval_one_hot = eval_one_hot
    
    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    im = self.data_imaging[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.default_transform(im)

    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data_tabular[index]))
    else:
      tab = torch.tensor(self.data_tabular[index], dtype=torch.float)

    label = torch.tensor(self.labels[index], dtype=torch.long)

    return (im, tab), label

  def __len__(self) -> int:
    return len(self.data_tabular)

class CelebaImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for evaluation.

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view is never augmented.
  """

  def __init__(
          self,
          data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float,
          data_path_tabular: str, eval_one_hot: bool,
          img_size: int, live_loading: bool, train: bool, target: str) -> None:

    # Imaging
    self.data_imaging = self.read_image_files(data_path_imaging)
    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      pass

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.field_lengths_tabular = torch.tensor([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=torch.int).cpu()

    self.eval_one_hot = eval_one_hot

    # Classifier
    self.labels = self.data_tabular['Attractive']
    self.id = self.data_tabular['img_index']  # Series
    self.data_tabular.drop(columns=['Attractive', 'img_index'], inplace=True)

    self.train = train

  def read_and_parse_csv(self, path_tabular: str) -> pd.DataFrame:
    """
    """
    data = pd.read_csv(path_tabular)
    con_cols = []
    cat_cols = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    for col in cat_cols:
      if data[col].max() > 100:
          data[col] = pd.cut(data[col], bins=20, labels=range(20))
    for col in con_cols:
      col_data = torch.tensor(data[col].values, dtype=torch.float)
      mean = torch.mean(col_data)
      std = torch.std(col_data)
      data[col] = torch.div(torch.sub(col_data, mean), std)
    return data

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

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table.
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int):
    pet_id = self.id.iloc[index]
    rand_img_idx = random.randint(1, len(self.data_imaging[pet_id])) - 1
    im = self.data_imaging[pet_id][rand_img_idx]
    if self.live_loading:
      im = Image.open(im).convert('RGB')

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.default_transform(im)

    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data_tabular.iloc[index].values, dtype=torch.float))
      tab = tab.float()
    else:
      tab = torch.tensor(self.data_tabular.iloc[index].values, dtype=torch.float)

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
    # one_hot = np.eye(2)
    # one_hot_label = one_hot[self.labels[index]]
    # label = torch.FloatTensor(one_hot_label)

    return (im, tab), label

  def __len__(self) -> int:
    return len(self.data_tabular)

class SunImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for evaluation.

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view is never augmented.
  """

  def __init__(
          self,
          data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float,
          data_path_tabular: str, eval_one_hot: bool,
          img_size: int, live_loading: bool, train: bool, target: str) -> None:

    # Imaging
    self.data_imaging = self.read_image_files(data_path_imaging)
    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      pass

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.field_lengths_tabular = torch.full((101,), 2, dtype=torch.int).cpu()

    self.eval_one_hot = eval_one_hot

    # Classifier
    self.labels = self.data_tabular['open area']
    self.id = self.data_tabular['image_path']  # Series
    self.data_tabular.drop(columns=['open area', 'image_path'], inplace=True)

    self.train = train

  def read_and_parse_csv(self, path_tabular: str) -> pd.DataFrame:
    """
    """
    data = pd.read_csv(path_tabular)
    for col in data.columns:
      if col == 'image_path' or col == 'open area':
        continue
      if data[col].max() > 100:
          data[col] = pd.cut(data[col], bins=20, labels=range(20))
    return data

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

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table.
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int):
    pet_id = self.id.iloc[index]
    rand_img_idx = random.randint(1, len(self.data_imaging[pet_id])) - 1
    im = self.data_imaging[pet_id][rand_img_idx]
    if self.live_loading:
      im = Image.open(im).convert('RGB')

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.default_transform(im)

    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data_tabular.iloc[index].values, dtype=torch.float))
      tab = tab.float()
    else:
      tab = torch.tensor(self.data_tabular.iloc[index].values, dtype=torch.float)

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
    # one_hot = np.eye(2)
    # one_hot_label = one_hot[self.labels.iloc[index]]
    # label = torch.FloatTensor(one_hot_label)

    return (im, tab), label

  def __len__(self) -> int:
    return len(self.data_tabular)