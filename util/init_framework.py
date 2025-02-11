from typing import Dict, Union
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import TensorDict
from tqdm import tqdm as tqdm
from tqdm.rich import tqdm as rich_tqdm
import platform


def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: th.device) -> Union[th.Tensor, TensorDict]:
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device,dtype=th.get_default_dtype())
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device,dtype=th.get_default_dtype()) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def preprocess_obs(
    obs: th.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.float() / 255.0
        return obs.to(th.get_default_dtype())

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.to(th.get_default_dtype())

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")



def origin_tqdm(self) -> None:
    step = self.locals["total_timesteps"] - self.model.num_timesteps
    step = 1 if step == 0 else step
    if platform.system() == 'Windows' and enable():
        self.pbar = rich_tqdm(total=step, desc=f"episode #{self.locals['total_timesteps'] // step}")
    else:
        self.pbar = tqdm(total=step,desc=f"episode #{self.locals['total_timesteps']//step}")


def enable():
    from ctypes import windll, wintypes, byref
    INVALID_HANDLE_VALUE = -1
    STD_OUTPUT_HANDLE = -11
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

    hOut = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    if hOut == INVALID_HANDLE_VALUE:
        return False
    dwMode = wintypes.DWORD()
    if windll.kernel32.GetConsoleMode(hOut, byref(dwMode)) == 0:
        return False
    dwMode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
    if windll.kernel32.SetConsoleMode(hOut, dwMode) == 0:
        return False
    return True

def to_torch(self,array: np.ndarray, copy: bool = True) -> th.Tensor:
    if copy:
        return th.tensor(array, device=self.device).to(th.get_default_dtype())
    return th.as_tensor(array, device=self.device).to(th.get_default_dtype())

def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
    last_values = last_values.clone().cpu().numpy().flatten()
    last_gae_lam = 0
    for step in reversed(range(self.buffer_size)):
        if step == self.buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_values = last_values
        else:
            next_non_terminal = 1.0 - self.episode_starts[step + 1]
            next_values = self.values[step + 1]
        delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
        last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        self.advantages[step] = last_gae_lam
    self.returns = self.advantages + self.values

def check(self, value):
    return torch.all(value >= 0, dim=-1)
