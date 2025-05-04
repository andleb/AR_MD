#!/usr/bin/env python
# coding: utf-8

# # Muller-Brown potential, BERT-style encoder testing - large data


import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import numpy as np

import joblib

# IMPORT HELPER FUNCTIONS
import torch

sys.path.append("../src/")
sys.path.append("../src/transformers/src/")

import modelling_md as modelling
import utils

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertConfig

# # Load MD data
#################################################################################s
# data = np.load("../data/MB/1000_nobeta.0.1.npz", allow_pickle=True)

# wilder data
data = np.load("../data/MB/1000_nobeta.0.0065.D100.npz", allow_pickle=True)
N = data["trajectories"].shape[0]

# Subsample data
# number of trajectories trained on
n_traj = 1000

idx = np.random.choice(N, n_traj, replace=False)

trajectories = data["trajectories"][idx]
states = data["labels"][idx]

seq_length = trajectories[0].shape[0]
n_features = trajectories[0].shape[1]
n_states = len(np.unique(states))

#################################################################################
batch_size = 6
num_epochs = 1000

# Create dataloader
dataset = modelling.MDTrajectoryDataset(trajectories=torch.tensor(trajectories),
                                        states=torch.tensor(states, dtype=torch.long))

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        # collate_fn=collate_trajectories
                        # num_workers=0   # Set to 0 for easier debugging
                        )

########################################################################
# ## Model & Train

#Small model
########################################################################
# prefix = 'models/MB_wilder_small'
# config = BertConfig(hidden_size=16,
#                     num_hidden_layers=4,
#                     num_attention_heads=8,
#                     intermediate_size=128,
#                     hidden_dropout_prob=0.1,
#                     attention_probs_dropout_prob=0.1,
#                     # NOTE: depends on traj length
#                     max_position_embeddings=1024,
#                     layer_norm_eps=1e-12)

# model = modelling.MDTrajectoryTransformer(config=config,  # NOTE: the space dimension
#                                           n_features=n_features,
#                                           n_states=n_states,
#                                           lr=1e-4,
#                                           weight_decay=0.01,
#                                           epochs=num_epochs,
#                                           warmup_epochs=25,  # NOTE: classification vs sequence loss weight
#                                           trajectory_loss_weight=1.,
#                                           state_loss_weight=1.)

########################################################################
# # # Normal model
# prefix = 'models/MB_wilder_normal'

# # Configure model
# config = BertConfig(hidden_size=64,
#                     num_hidden_layers=8,
#                     num_attention_heads=16,
#                     intermediate_size=256,
#                     hidden_dropout_prob=0.1,
#                     attention_probs_dropout_prob=0.1,
#                     max_position_embeddings=1024,
#                     layer_norm_eps=1e-12)

# model = modelling.MDTrajectoryTransformer(config=config,  # NOTE: the space dimension
#                                           n_features=n_features,
#                                           n_states=n_states,
#                                           lr=1e-4,
#                                           weight_decay=0.01,
#                                           epochs=num_epochs,
#                                           warmup_epochs=50,  
#                                           # NOTE: classification vs sequence loss weight
#                                           trajectory_loss_weight=1.,
#                                           state_loss_weight=1.)

#########################################################################################################
# Huge model:
prefix = 'models/MB_wilder_large'

# Configure model
config = BertConfig(
    hidden_size=120,          # Increased
    num_hidden_layers=12,     # Increased
    num_attention_heads=24,
    intermediate_size=512,   # Increased
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=1024,
    layer_norm_eps=1e-12
)

model = modelling.MDTrajectoryTransformer(config=config,  # NOTE: the space dimension
                                          n_features=n_features,
                                          n_states=n_states,
                                          lr=1e-4,
                                          weight_decay=0.01,
                                          epochs=num_epochs,
                                          warmup_epochs=50,  # NOTE: classification vs sequence loss weight
                                          trajectory_loss_weight=1.,
                                          state_loss_weight=1.)


#########################################################################################################
os.makedirs(prefix, exist_ok=True)
joblib.dump(config, f"{prefix}/config.jblb")

loss_tracker = utils.LossTracker()

# trainer = pl.Trainer(
#     max_epochs= num_epochs,
#     gradient_clip_val=1.0,
#     log_every_n_steps=100,
#     accelerator="gpu",

#     devices=1,

#     # devices=[1],

#     # devices="auto",
#     # strategy="ddp_spawn",

#     # devices=[0, 1],  # Explicitly use GPUs 0 and 1
#     # strategy="ddp_spawn",

#     # gradient_checkpointing=True,  # Add this to reduce memory usage    
#     # port=find_free_port()  # Use a random free port

#     # auto_scale_batch_size="binsearch",  # Let Lightning find optimal batch size

#     callbacks=[
#        loss_tracker,
#         pl.callbacks.ModelCheckpoint(
#             dirpath=prefix,
#             filename='model-{epoch:04d}',
#             every_n_epochs=100,
#             save_top_k=-1,  # Save all models
#             save_last=True  # Additionally save as 'last.ckpt'
#         ),
#         pl.callbacks.LearningRateMonitor(logging_interval='step'),
#     ],
# )


trainer = pl.Trainer(
                     max_epochs=num_epochs,
    
                     # resume_from_checkpoint=f"{prefix}/last-v2.ckpt/",
                     
                     gradient_clip_val=1.0,
                     log_every_n_steps=100,

                     accelerator="gpu",
                     devices="auto",

                     # Use DeepSpeed for model parallelism
                     strategy="deepspeed_stage_2",
                     precision=16,

                     auto_scale_batch_size="binsearch",  # Let Lightning find optimal batch size

                     callbacks=[loss_tracker,
                                pl.callbacks.ModelCheckpoint(
                                                                           dirpath=prefix,
                                                                           filename='model-{epoch:04d}',
                                                                           every_n_epochs=100,
                                                                           # Save all models
                                                                           save_top_k=-1,
                                                                           save_last=True
                                                                           ),
                                pl.callbacks.LearningRateMonitor(logging_interval='step'),
                                ],
                     )

#########################################################################################################

trainer.fit(model, dataloader)


#########################################################################################################
joblib.dump(loss_tracker.training_losses, f"{prefix}/training_losses.jblb")
