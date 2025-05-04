import time
from typing import *

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
)
from transformers.optimization import get_linear_schedule_with_warmup


class MDTrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[torch.Tensor], states: Optional[List[int]] = None):
        # self.trajectories = trajectories
        self.states = states

        # Normalize trajectories
        self.means = [traj.mean(dim=0) for traj in trajectories]
        self.stds = [traj.std(dim=0) for traj in trajectories]
        self.normalized_trajectories = [
            (traj - mean) / (std + 1e-8)
            for traj, mean, std in zip(trajectories, self.means, self.stds)
        ]
        self.lengths = [len(traj) for traj in trajectories]

    def __len__(self):
        return len(self.normalized_trajectories)

    def __getitem__(self, idx):
        # Return a single trajectory without batch dimension - DataLoader will add it
        normalized_traj = self.normalized_trajectories[idx]  # [seq_len, features]
        output = {
            'trajectories': normalized_traj,
            'lengths'     : torch.tensor([self.lengths[idx]]),
            'means'       : self.means[idx],
            'stds'        : self.stds[idx]
        }

        if self.states is not None:
            output['states'] = torch.tensor([self.states[idx]], dtype=torch.long)

        return output


# Collate function to handle variable-length trajectories
def collate_trajectories(batch):
    trajectories = []
    lengths = []
    for item in batch:
        trajectories.extend(item['trajectories'])
        lengths.extend(item['lengths'])
    return {
        'trajectories': trajectories,
        'lengths'     : torch.tensor(lengths)
    }


# NOTE: This is just an MLP that maps back to the coordinate space; much like for angles in foldingdiff
class FeaturePredictor(nn.Module):
    """Projects transformer outputs back to feature space."""

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dense2 = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


# NOTE: in case we have labeled data
class StateClassifier(nn.Module):
    """Classifies states based on trajectory embeddings."""

    def __init__(self, d_model: int, n_states: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dense2 = nn.Linear(d_model, n_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_length, d_model]
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")

        x = x.mean(dim=1)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


class MDTrajectoryTransformerBase(BertPreTrainedModel):
    """Base transformer model with both trajectory prediction and state classification."""

    def __init__(
            self,
            config,
            n_features: int,
            n_states: int = 0,  # Default to 0 to disable state prediction
    ) -> None:
        super().__init__(config)
        self.config = config
        self.n_features = n_features
        self.n_states = n_states

        # Project features to hidden dimension
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=n_features,
            out_features=config.hidden_size
        )

        # Use built-in BERT embeddings which include position embeddings
        self.embeddings = self.get_embeddings()

        # Transformer encoder
        self.encoder = BertEncoder(config)

        # Two heads:
        # Trajectory prediction head
        self.feature_predictor = FeaturePredictor(
            config.hidden_size,
            n_features
        )

        # Optional state classification head
        self.state_classifier = StateClassifier(
            config.hidden_size,
            n_states
        ) if n_states >= 2 else None

        self.init_weights()

    def get_embeddings(self):
        """Creates embedding layer - separated for easy subclassing"""
        embeddings = nn.ModuleDict({
            'position_embeddings': nn.Embedding(
                self.config.max_position_embeddings,
                self.config.hidden_size
            ),
            'LayerNorm'          : nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
            'dropout'            : nn.Dropout(self.config.hidden_dropout_prob)
        })
        return embeddings

    def forward(
            self,
            features: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_embeddings: bool = False,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with option to output attention patterns"""
        # Add batch dimension if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(0)

        batch_size, seq_length, n_features = features.shape
        assert n_features == self.n_features, f"Expected {self.n_features} features, got {n_features}"

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=features.device)
            position_ids = position_ids.expand(batch_size, -1)

        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=features.device)

        # Create extended attention mask for transformer
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Cast features to model dtype
        features = features.to(dtype=self.encoder.layer[0].attention.self.query.weight.dtype)
        extended_attention_mask = extended_attention_mask.to(dtype=features.dtype)

        # Project features to hidden dimension
        hidden_states = self.inputs_to_hidden_dim(features)

        # Add positional embeddings
        position_embeddings = self.embeddings['position_embeddings'](position_ids)
        position_embeddings = position_embeddings.to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.embeddings['LayerNorm'](hidden_states)
        hidden_states = self.embeddings['dropout'](hidden_states)

        # Pass through transformer encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get predictions
        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        trajectory_pred = self.feature_predictor(sequence_output)

        if return_dict:
            outputs = {
                'trajectory'       : trajectory_pred,
                'last_hidden_state': sequence_output,
            }
            if output_attentions:
                outputs['attentions'] = encoder_outputs.attentions
            if output_hidden_states:
                outputs['hidden_states'] = encoder_outputs.hidden_states
            if self.state_classifier is not None:
                outputs['state'] = self.state_classifier(sequence_output)
            if return_embeddings:
                outputs['embeddings'] = sequence_output

            return outputs

        return trajectory_pred


class MDTrajectoryTransformer(MDTrajectoryTransformerBase, pl.LightningModule):
    def __init__(
            self,
            config,
            n_features: int,
            n_states: int = 0,
            lr: float = 1e-4,
            weight_decay: float = 0.0,
            epochs: int = 100,
            warmup_epochs: int = 10,
            trajectory_loss_weight: float = 1.0,
            state_loss_weight: float = 1.0,
    ):

        pl.LightningModule.__init__(self)

        # Manually set hparams
        self.hparams['config'] = config
        self.hparams['n_features'] = n_features
        self.hparams['n_states'] = n_states
        self.hparams['lr'] = lr
        self.hparams['weight_decay'] = weight_decay
        self.hparams['epochs'] = epochs
        self.hparams['warmup_epochs'] = warmup_epochs
        self.hparams['trajectory_loss_weight'] = trajectory_loss_weight
        self.hparams['state_loss_weight'] = state_loss_weight

        MDTrajectoryTransformerBase.__init__(
            self,
            config=config,
            n_features=n_features,
            n_states=n_states
        )

        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.trajectory_loss_weight = trajectory_loss_weight
        self.state_loss_weight = state_loss_weight if n_states >= 2 else 0.0

        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    def _compute_loss(self, batch, split: str = "train"):
        """Compute loss with optional state prediction."""
        trajectories = batch['trajectories']
        lengths = batch['lengths']

        # Skip empty batches
        if trajectories.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Ensure trajectories have the right shape
        if len(trajectories.shape) > 3:
            trajectories = trajectories.squeeze(1)

        # Cast trajectories to model dtype
        model_dtype = self.encoder.layer[0].attention.self.query.weight.dtype
        trajectories = trajectories.to(dtype=model_dtype)

        # Get predictions
        # Predict all but last frame
        outputs = self.forward(trajectories[:, :-1])
        trajectory_pred = outputs['trajectory']

        # Compute trajectory loss
        trajectory_loss = F.mse_loss(trajectory_pred, trajectories[:, 1:])
        combined_loss = self.trajectory_loss_weight * trajectory_loss

        # Log the trajectory loss
        self.log(f"{split}_trajectory_loss", trajectory_loss, sync_dist=True)

        # Compute state loss if enabled
        if self.state_classifier is not None and 'states' in batch:
            state_pred = outputs['state']
            states = batch['states'].reshape(-1)
            state_loss = F.cross_entropy(state_pred, states)
            combined_loss += self.state_loss_weight * state_loss

            accuracy = (state_pred.argmax(dim=-1) == states).float().mean()

            self.log(f"{split}_state_loss", state_loss, sync_dist=True)
            self.log(f"{split}_accuracy", accuracy, sync_dist=True)

        self.log(f"{split}_loss", combined_loss, sync_dist=True)
        return combined_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "val")
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_epochs,
            num_training_steps=self.epochs
        )

        return {
            "optimizer"   : optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval" : "epoch"
            }
        }

    def training_epoch_end(self, outputs) -> None:
        losses = torch.stack([o["loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([o["val_loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        pl.utilities.rank_zero_info(
            f"Valid loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}"
        )

    @torch.no_grad()
    def predict_trajectory(
            self,
            initial_frames: torch.Tensor,
            n_steps: int,
            means: Optional[torch.Tensor] = None,
            stds: Optional[torch.Tensor] = None,
            stride: int = 1,
    ):
        """
        Predict multiple trajectories simultaneously

        Args:
            initial_frames: Raw (unnormalized) initial frames
            n_steps: Number of steps to predict
            means: Optional means for each trajectory
            stds: Optional stds for each trajectory
            stride: How many steps to predict at once
        """
        device = initial_frames.device
        n_traj, seq_length, n_features = initial_frames.shape

        if means is None:
            means = initial_frames.mean(dim=1)
        if stds is None:
            stds = initial_frames.std(dim=1)

        # Normalize initial frames
        means_expanded = means.unsqueeze(1)
        stds_expanded = stds.unsqueeze(1)
        initial_frames_norm = (initial_frames - means_expanded) / (stds_expanded + 1e-8)

        # Initialize trajectory storage with normalized initial frames
        trajectory_norm = torch.zeros(n_traj, seq_length + n_steps, n_features).to(device)
        trajectory_norm[:, :seq_length] = initial_frames_norm

        # Generate trajectories using sliding window
        current_sequence = initial_frames_norm
        for t in range(0, n_steps, stride):
            # Predict next frames (will be normalized since model trained on normalized data)
            predictions = self.forward(current_sequence)["trajectory"]

            # Take only the needed predictions
            next_frames = predictions[:, -stride:]

            # Add to normalized trajectory
            end_idx = min(seq_length + t + stride, trajectory_norm.size(1))
            trajectory_norm[:, seq_length + t:end_idx] = next_frames[:, :end_idx - (seq_length + t)]

            # Update current sequence by sliding window
            current_sequence = trajectory_norm[:, t + 1:seq_length + t + 1]

        # Un-normalize the trajectories
        means_expanded = means.unsqueeze(1)  # [n_traj, 1, n_features]
        stds_expanded = stds.unsqueeze(1)  # [n_traj, 1, n_features]
        trajectories = trajectory_norm * stds_expanded + means_expanded

        return trajectories

    @torch.no_grad()
    def classify_state(
            self,
            trajectories: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Classify the state of one or more trajectories.

        Args:
            trajectories: Single trajectory tensor or list of trajectory tensors

        Returns:
            Dictionary containing:
                'logits': Raw classification logits [n_trajectories, n_states]
                'probabilities': Softmax probabilities [n_trajectories, n_states]
                'predictions': Predicted class indices [n_trajectories]
        """
        if self.state_classifier is None:
            raise RuntimeError("State classification is not enabled (n_states < 2)")

        if isinstance(trajectories, torch.Tensor) and len(trajectories.shape) == 2:
            trajectories = [trajectories]

        # Normalize each trajectory
        normalized_trajectories = []
        for traj in trajectories:
            mean = traj.mean(dim=0)
            std = traj.std(dim=0)
            traj_norm = (traj - mean) / (std + 1e-8)
            normalized_trajectories.append(traj_norm)

        # Get predictions for each trajectory
        all_logits = []
        for traj in normalized_trajectories:
            outputs = self.forward(traj)
            all_logits.append(outputs['state'])

        state_logits = torch.stack(all_logits)
        probabilities = F.softmax(state_logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return {
            'logits'       : state_logits,
            'probabilities': probabilities,
            'predictions'  : predictions
        }
