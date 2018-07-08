from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util.dataloader import DataLoader
from model.network import AdaptReID
from dataset.duke import Duke
import config
