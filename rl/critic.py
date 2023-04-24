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
import linecache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from transformers import ViltProcessor, ViltModel, ViltConfig


def set_init(layers):
  for layer in layers:
    nn.init.normal_(layer.weight, mean=0., std=0.05)
    nn.init.constant_(layer.bias, 0.)


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim, task_dim, max_action, params):
    super(Critic, self).__init__()
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
    # for i, param in enumerate(self.vilt_model.parameters()):
    #   if i<=105:
    #     param.requires_grad = False

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

    self.action_feat_block1 = nn.Linear(49 * 7 + 7, 256)

    self.action_feat_block2 = nn.Linear(256, 256)
    self.action_feat_block3 = nn.Linear(256, 128)

    self.critic_feat_block1 = nn.Linear(256 + 128 + 128, 512)
    self.critic_feat_block2 = nn.Linear(512, 128)
    self.critic_feat_block3 = nn.Linear(128, 64)
    self.critic_feat_block4 = nn.Linear(64, 16)
    self.critic_feat_block5 = nn.Linear(16, 1)

    set_init([self.img_feat_block2, self.task_feat_block1, self.task_feat_block2, self.task_feat_block3,\
      self.action_feat_block1, self.action_feat_block2, self.action_feat_block3,\
      self.critic_feat_block1, self.critic_feat_block2, self.critic_feat_block3, self.critic_feat_block4,\
      self.critic_feat_block5])

  def forward(self, state, task_vec, action):
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

    action_feat = F.relu(self.action_feat_block1(action))
    action_feat = F.relu(self.action_feat_block2(action_feat))
    action_feat = F.relu(self.action_feat_block3(action_feat))

    critic_feat = torch.cat([action_feat_raw, action_feat], -1)
    critic_feat = F.relu(self.critic_feat_block1(critic_feat))
    critic_feat = F.relu(self.critic_feat_block2(critic_feat))
    critic_feat = F.relu(self.critic_feat_block3(critic_feat))
    critic_feat = F.relu(self.critic_feat_block4(critic_feat))
    q_a = self.critic_feat_block5(critic_feat)
    return q_a
