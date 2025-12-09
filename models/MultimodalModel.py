import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel

class MultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(MultimodalModel, self).__init__()

    self.imaging_model = ImagingModel(args)
    self.tabular_model = TabularModel(args)
    # self.cat_list = [2] * 39
    # self.tabular_model = TabularTransformerEncoder(self.cat_list, [])
    in_dim = 2048
    # self.i2t = nn.Linear(2048, 512)
    self.head = nn.Linear(in_dim, args.num_classes)
    # self.head1 = nn.Linear(in_dim, args.num_classes)

  def forward(self, x: torch.Tensor):
    x_im = self.imaging_model.encoder(x[0])[0].squeeze()
    # x_im = self.i2t(x_im)
    x_tab = self.tabular_model.encoder(x[1]).squeeze() #.squeeze() #[:,0,:]
    # x = torch.cat([x_im, x_tab], dim=1)
    # x = self.head(x)
    # x_im = self.head(x_im_f)
    # x_tab = self.head1(x_tab_f)

    # x = torch.stack([x_im, x_tab], dim=1)
    # x, _ = torch.max(x, dim=1)
    # x = self.head(x)
    return x_im, x_tab

  def forward_image(self, x):
    x_im = self.imaging_model.encoder(x)[0].squeeze()
    # x_im = self.i2t(x_im)
    return x_im

  def forward_table(self, x):
    x_tab = self.tabular_model.encoder(x).squeeze()  #.squeeze() #[:,0,:]
    return x_tab