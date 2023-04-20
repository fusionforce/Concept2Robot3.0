import copy
import numpy as np
import os
import sys
import time
np.set_printoptions(precision=4,suppress=False)

import importlib
import glob
import imageio
import math
import datetime
from PIL import Image
import linecache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from transformers import ViltProcessor, ViltModel, ViltConfig


def set_init(layers):
  for layer in layers:
    nn.init.normal_(layer.weight, mean=0., std=0.05)
    nn.init.constant_(layer.bias, 0.)

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, task_dim, max_action, params):
    super(Actor, self).__init__()
    self.params = params
    self.model = models.resnet18(pretrained=True)
    self.action_dim = action_dim
    self.max_action = max_action
    self.raw_text = eval(linecache.getline('../Languages/labels.txt', self.params.task_id+1).strip().split(":")[0])
    self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-2])
    # ViLT
    vilt_config = ViltConfig()
    vilt_config.hidden_size = 64
    vilt_config.num_hidden_layers = 8
    vilt_config.num_attention_heads = 8

    self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    self.vilt_model = ViltModel(vilt_config).from_pretrained("dandelin/vilt-b32-mlm")

    # Freeze layers except last 2
    for i, param in enumerate(self.vilt_model.parameters()):
      if i<=105:
        param.requires_grad = False

    self.vilt_layers = nn.Sequential(
            nn.Linear(197*768, 768),
            nn.Linear(768, 384)
    )
    self.img_feat_block1 = nn.Sequential(
      nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(256),
    )
    self.img_feat_block2 = nn.Linear(256 * 2 * 3, 256)

    self.task_feat_block1 = nn.Linear(1024, 512)
    self.task_feat_block2 = nn.Linear(512, 256)
    self.task_feat_block3 = nn.Linear(256, 128)

    self.action_feat_block1 = nn.Linear(256 + 128, 256)
    self.action_feat_block2 = nn.Linear(256, 128)
    self.action_feat_block3 = nn.Linear(128, 64)
    self.action_feat_block4 = nn.Linear(64, 7)

    #####3 Force
    # 1
    self.force_feat_block1 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256 + 128, out_channels=256, kernel_size=4, stride=1, bias=True),
      nn.ReLU(),
    )

    # 3
    self.force_feat_block2 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
      nn.ReLU(),
    )

    # 7
    self.force_feat_block3 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
      nn.ReLU(),
    )

    #
    self.force_feat_block4 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
      nn.ReLU(),
    )

    self.force_feat_block5 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=64, out_channels=self.params.a_dim, kernel_size=3, stride=2, padding=1),
    )

    set_init([self.img_feat_block2, self.task_feat_block1, self.task_feat_block2, self.task_feat_block3,\
      self.action_feat_block1, self.action_feat_block2, self.action_feat_block3, \
      self.action_feat_block4])

  def forward(self, state, task_vec, training=False):
    pil_list = []
    for i in range(state.shape[0]):
      pil_list.append(Image.fromarray(state[i]))
    text_list = []
    for i in range(state.shape[0]):
      text_list.append(self.raw_text)
    inputs = self.vilt_processor(pil_list, text_list, return_tensors="pt")
    for key in inputs:
      inputs[key] = inputs[key].cuda()
    outputs = self.vilt_model(**inputs, return_dict=True, output_hidden_states=True)
    last_hidden_states = outputs.last_hidden_state
    action_feat_raw = self.vilt_layers(last_hidden_states.view(state.shape[0], -1))

    ### generate goal
    action_feat = F.relu(self.action_feat_block1(action_feat_raw))
    action_feat = F.relu(self.action_feat_block2(action_feat))
    action_feat = F.relu(self.action_feat_block3(action_feat))
    weights_goal = np.ones((1,7))
    weights_goal[0,:3] *= self.max_action
    weights_goal[0,3:6] *= self.params.rotation_max_action
    weights_goal = torch.FloatTensor(weights_goal).to("cuda")
    goal = torch.tanh(self.action_feat_block4(action_feat)) * weights_goal
    #transl_action = torch.tanh(self.action_feat_block4(action_feat)) * self.max_action
    #rot_action = torch.tanh(self.action_feat_block5(action_feat)) * self.params.rotation_max_action
    #gripper_action = torch.tanh(self.action_feat_block6(action_feat))
    #goal = torch.cat([transl_action, rot_action, gripper_action],axis=-1)

    ### generate force
    force_feat = action_feat_raw.unsqueeze(2)
    force_feat = F.relu(self.force_feat_block1(force_feat))
    force_feat = F.relu(self.force_feat_block2(force_feat))
    force_feat = F.relu(self.force_feat_block3(force_feat))
    force_feat = F.relu(self.force_feat_block4(force_feat))
    force_feat = torch.tanh(self.force_feat_block5(force_feat))
    _, n_dim, timesteps = force_feat.size()
    force = torch.transpose(force_feat, 1, 2)
    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((1, self.params.traj_timesteps, 1)) * float(
      self.params.traj_timesteps)
    weights = torch.FloatTensor(weights).to("cuda")
    force = weights * force

    if training:
      force = force.reshape((bs,-1))
      action = torch.cat([goal, force],axis=-1)
      return action
    else:
      return goal, force