import argparse
import os
import time
import types

import stable_baselines3.common.utils
import torch.distributions.constraints
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import get_latest_run_id, configure_logger
from stable_baselines3.common.vec_env import SubprocVecEnv

from util import InstanceParameter
from Env import CTEnv
from network import *
from util.init_framework import *

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='PPO') # PPO or DQN or DDQN
    parser.add_argument('--strength', type=int, default=2)
    parser.add_argument('--net', type=list, default=[512,512,512,512,512])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step', type=int, default=2048) # PPO
    parser.add_argument('--train-freq', type=int, default=16)  # DQN and DDQN
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.65)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=16)
    parser.add_argument('--optim', type=object, default=th.optim.RMSprop)
    parser.add_argument('--activation-fn', type=object, default=th.nn.ReLU)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_known_args()[0]
    return args



def run(args):
    print(arg.name)
    print(device)
    th.set_default_dtype(th.double)
    train_envs = SubprocVecEnv([lambda:Monitor(CTEnv(name=args.name, t_way=args.strength)) for _ in range(args.training_num)])
    if args.model_name == "PPO":
        policy_kwargs = dict(activation_fn=args.activation_fn,
                             net_arch=dict(pi=args.net, vf=args.net),
                             optimizer_class=args.optim)
        model = CTMaskablePPO(
            policy="MlpPolicy",
            env=train_envs,
            learning_rate=args.lr,
            n_steps=args.step//args.training_num,
            gamma=args.gamma,
            n_epochs=args.epoch,
            device=device,
            batch_size=args.batch_size,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif args.model_name == "DQN":
        policy_kwargs = dict(activation_fn=args.activation_fn,
                             net_arch=args.net,
                             optimizer_class=args.optim)
        model = CTMaskableDQN(
            env=train_envs,
            learning_rate=args.lr,
            gamma=args.gamma,
            train_freq=args.train_freq//args.training_num,
            gradient_steps=args.epoch,
            device=device,
            batch_size=args.batch_size,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif args.model_name == "DDQN":
        policy_kwargs = dict(activation_fn=args.activation_fn,
                             net_arch=args.net,
                             optimizer_class=args.optim)
        model = CTMaskableDDQN(
            env=train_envs,
            learning_rate=args.lr,
            gamma=args.gamma,
            train_freq=args.train_freq//args.training_num,
            gradient_steps=args.epoch,
            device=device,
            batch_size=args.batch_size,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    # init
    pb_callback = ProgressBarCallback()
    pb_callback._on_training_start = types.MethodType(origin_tqdm, pb_callback)
    stable_baselines3.common.utils.obs_as_tensor.__code__ = obs_as_tensor.__code__
    stable_baselines3.common.preprocessing.preprocess_obs.__code__ = preprocess_obs.__code__
    torch.distributions.constraints._Simplex.check.__code__ = check.__code__
    # init end
    last_run_id = get_latest_run_id(args.logdir+"/"+args.model_name+"/", args.name)+1
    logger = configure_logger(verbose=0, tensorboard_log=args.logdir, tb_log_name=args.name, reset_num_timesteps=True)
    logger.record("args",str(args))
    model.set_logger(logger)
    model_name = "model/"+args.model_name+"/" + args.name + "/"+args.model_name+"_" + str(last_run_id)
    start_time = time.time()
    model.learn(
        total_timesteps=InstanceParameter.prog_step[args.name],
        reset_num_timesteps=False,
        callback=pb_callback
    )
    train_time = time.time() - start_time
    model.save(model_name)
    print(f"Training time: {train_time}")
    path = "results/train_time/"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"results/train_time/{args.model_name}.txt", "a") as f:
        f.write(f"{last_run_id}:{train_time}\n")



if __name__ == '__main__':
    for name in InstanceParameter.prog_param:
        arg = get_args()
        arg.name = name
        run(arg)
