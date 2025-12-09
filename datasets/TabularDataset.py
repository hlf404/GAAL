from typing import Tuple
import csv
import copy
import pandas as pd
import glob
import os
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
  """"
  Dataset for the evaluation of tabular data
  """
  def __init__(self, data_path: str, labels_path: str, eval_one_hot: bool=True, field_lengths_tabular: str=None):
    super(TabularDataset, self).__init__()
    self.data = self.read_and_parse_csv(data_path)
    self.labels = torch.load(labels_path)
    self.eval_one_hot = eval_one_hot
    self.field_lengths = torch.load(field_lengths_tabular)

    if self.eval_one_hot:
      for i in range(len(self.data)):
        self.data[i] = self.one_hot_encode(torch.tensor(self.data[i]))
    else:
      self.data = torch.tensor(self.data, dtype=torch.float)

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.data[0])
  
  def read_and_parse_csv(self, path: str):
    """
    Does what it says on the box
    """
    with open(path,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.data[index], self.labels[index]

  def __len__(self) -> int:
    return len(self.data)

class CelebaTabularDataset(Dataset):
  """"
  Dataset for the evaluation of tabular data
  """

  def __init__(self, data_path: str, eval_one_hot: bool = True):
    super(CelebaTabularDataset, self).__init__()
    self.data = self.read_and_parse_csv(data_path)
    self.labels = self.data['Attractive']
    self.data.drop(columns=['Attractive', 'img_index'], inplace=True)
    self.eval_one_hot = eval_one_hot
    self.field_lengths = torch.tensor([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=torch.int).cpu()


  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table.
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.data[0])

  def read_and_parse_csv(self, path_tabular: str) -> pd.DataFrame:
    """
    """
    data = pd.read_csv(path_tabular)
    con_cols = []
    cat_cols = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                'Mouth_Slightly_Open',
                'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
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

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i], min=0, max=self.field_lengths[i] - 1).long(),
                                               num_classes=int(self.field_lengths[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data.iloc[index].values, dtype=torch.float))
      tab = tab.float()
    else:
      tab = torch.tensor(self.data.iloc[index].values, dtype=torch.float)

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)

    return tab, label

  def __len__(self) -> int:
    return len(self.data)

class SunTabularDataset(Dataset):
  """"
  Dataset for the evaluation of tabular data
  """

  def __init__(self, data_path: str, eval_one_hot: bool = True):
    super(SunTabularDataset, self).__init__()
    self.data = self.read_and_parse_csv(data_path)
    self.labels = self.data['open area']
    self.data.drop(columns=['open area', 'image_path', 'semi-enclosed area', 'enclosed area'], inplace=True)
    self.eval_one_hot = eval_one_hot
    self.field_lengths = torch.full((99,), 2, dtype=torch.int).cpu()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table.
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.data[0])

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

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i], min=0, max=self.field_lengths[i] - 1).long(),
                                               num_classes=int(self.field_lengths[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data.iloc[index].values, dtype=torch.float))
      tab = tab.float()
    else:
      tab = torch.tensor(self.data.iloc[index].values, dtype=torch.float)

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)

    return tab, label

  def __len__(self) -> int:
    return len(self.data)