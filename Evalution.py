import os
import time

import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space

from Env import CTEnv
from network import *
from util import InstanceParameter
from util.init_framework import *


def eval_model(name,model_name, index,record_num):
    start_time = time.perf_counter_ns()
    torch.set_default_dtype(torch.double)
    env = CTEnv(name=name, t_way=2)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(device)
    if model_name == "PPO":
        model = CTMaskablePPO.load(f"model/{model_name}/{name}/{model_name}_{index}.zip", env=env, device=device)
    elif model_name == "DQN":
        model = CTMaskableDQN.load(f"model/{model_name}/{name}/{model_name}_{index}.zip", env=env, device=device)
    elif model_name == "DDQN":
        model = CTMaskableDDQN.load(f"model/{model_name}/{name}/{model_name}_{index}.zip", env=env, device=device)

    stable_baselines3.common.utils.obs_as_tensor.__code__ = obs_as_tensor.__code__
    stable_baselines3.common.preprocessing.preprocess_obs.__code__ = preprocess_obs.__code__
    path = f"results/test_suite/{model_name}/"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    file = open(path+f"{name}.txt", "w")
    episode = 0
    obs,_ = env.reset()
    while episode < record_num:
        action_masks = env.action_masks()
        action, obs = model.predict(obs, action_masks=action_masks,deterministic=False)
        obs, reward, done,truncate, info = env.step(action)
        if done:
            gen_time = time.perf_counter_ns() - start_time
            file.write(f'{len(env.tc_set)}\t{gen_time}\n')
            file.flush()
            # init
            start_time = time.perf_counter_ns()
            obs, _ = env.reset()
            episode += 1
    file.close()



if __name__ == '__main__':
    record_num = 1000
    model_name = "PPO" # DQN,DDQN or PPO
    model_index = {
        "inst1": 1,
        "inst2": 1,
        "inst3": 1,
        "inst4": 1,
        "inst5": 1,
        'inst6': 1,
        'inst7': 1,
        "inst8": 1,
        'inst9': 1,
        'inst10': 1,
        'inst11': 1,
        'inst12': 1,
    }
    for name in InstanceParameter.prog_param:
        eval_model(name,model_name,model_index[name],record_num)