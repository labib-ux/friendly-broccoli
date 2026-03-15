import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom neural network feature extractor for the RL policy.

    This extractor compresses the environment observation space into a dense
    feature representation. It strictly uses LayerNorm over BatchNorm to ensure
    stability during live trading (batch size of 1) and prevents gradient
    explosion during out-of-distribution market volatility.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128
    ) -> None:
        """
        Initialize the CustomFeatureExtractor.

        Args:
            observation_space: The Gymnasium observation space of the environment.
            features_dim: The dimensionality of the extracted features array.
        """
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            observations: A batch of observations from the environment.

        Returns:
            The extracted feature tensor of size (batch_size, features_dim).
        """
        return self.network(observations)
