import os
import logging
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from agent.policy import CustomFeatureExtractor
from config import MODEL_SAVE_PATH

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    Reinforcement learning agent wrapper using PPO for trading securely.

    This class encapsulates model initialization, device selection logic,
    the training loop, inference prediction, and safe saving/loading of the
    underlying Stable Baselines 3 model.
    """

    def __init__(
        self,
        env=None,
        tensorboard_log: str = "./tensorboard_logs/"
    ) -> None:
        """
        Initialize the TradingAgent.

        Args:
            env: The Gymnasium compatible training environment.
            tensorboard_log: The directory to store Tensorboard logs.

        Raises:
            Exception: If model initialization fails irreparably.
        """
        self.env = env
        self.logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "cpu"
            self.logger.warning(
                "Apple MPS detected but defaulting to CPU. "
                "MPS has known instability with stable-baselines3 PPO "
                "advantage calculation. Pass device='mps' manually to override."
            )
        else:
            self.device = "cpu"
            
        self.logger.info("Training device selected: %s", self.device)

        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )

        try:
            if os.path.exists(MODEL_SAVE_PATH) and os.path.isfile(MODEL_SAVE_PATH):
                self.model = PPO.load(
                    MODEL_SAVE_PATH,
                    env=self.env,
                    device=self.device,
                    tensorboard_log=tensorboard_log
                )
                self.logger.info("Loaded existing model from: %s", MODEL_SAVE_PATH)
            else:
                if os.path.exists(MODEL_SAVE_PATH) and not os.path.isfile(MODEL_SAVE_PATH):
                    self.logger.warning(
                        "MODEL_SAVE_PATH '%s' exists but is a directory, not a file. "
                        "Initializing new model instead.", MODEL_SAVE_PATH
                    )
                self.model = PPO(
                    "MlpPolicy",
                    env=self.env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    ent_coef=0.01,
                    clip_range=0.2,
                    verbose=0,
                    device=self.device,
                    tensorboard_log=tensorboard_log
                )
                self.logger.info("New model initialized.")
        except Exception as e:
            self.logger.error("Failed to initialize or load PPO model.", exc_info=True)
            raise

    def train(
        self,
        total_timesteps: int = 100000,
        reset_num_timesteps: bool = True
    ) -> None:
        """
        Train the RL agent within the provided environment.

        reset_num_timesteps usage:
            True  = fresh training run, timestep counter resets to 0,
                    learning rate schedule starts from beginning.
                    Use for initial training.
            False = continued training, timestep counter accumulates,
                    TensorBoard graphs continue from previous run.
                    Use explicitly when resuming — do not default to False
                    or learning rate will be incorrectly decayed from step 1.

        Args:
            total_timesteps: The total number of environment steps to train for.
            reset_num_timesteps: Whether to reset the training step counter to zero.

        Raises:
            ValueError: If the environment was not provided at instantiation.
            Exception: If training fails during the PPO learning phase.
        """
        if self.env is None:
            raise ValueError("Cannot train: no environment provided to TradingAgent.")

        self.logger.info(
            "Starting training for %d timesteps. reset_num_timesteps=%s",
            total_timesteps, reset_num_timesteps
        )

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=True
            )
            self.save()
        except Exception as e:
            self.logger.error("Training failed at step: %s", e, exc_info=True)
            raise

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict an action based on the observation.

        Args:
            observation: State tensor returned by the environment step/reset.
            deterministic: Whether to take highest probability action globally.

        Returns:
            The selected action as an integer constraint, defaulting to 0 (HOLD).
        """
        try:
            action_array, _state = self.model.predict(
                observation, deterministic=deterministic
            )
            return int(action_array.flat[0])
        except Exception as e:
            self.logger.error(
                "predict() failed: %s — returning HOLD (0) as safe fallback",
                e, exc_info=True
            )
            return 0  # HOLD is always the safest fallback — no position change

    def save(self) -> None:
        """
        Save the PPO agent model to disk safely.
        """
        try:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            self.model.save(MODEL_SAVE_PATH)
            self.logger.info("Model saved successfully to: %s", MODEL_SAVE_PATH)
        except Exception as e:
            self.logger.error("Model save failed: %s", e, exc_info=True)
