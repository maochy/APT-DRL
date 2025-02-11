import copy
import math
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from util import InstanceParameter


def get_vc_num(param, PC):
    count = 1
    for i in PC:
        count *= param[i]
    return count

def get_all_tuple(param, t_way):
    PC = []
    VC = []
    starts = np.zeros(t_way, dtype=int)
    depth = 0
    pair = np.zeros(t_way, dtype=int)
    vc_len = 0
    while starts[0] <= len(param) - t_way:
        if depth == t_way - 1:
            while starts[depth] < len(param):
                pair[depth] = starts[depth]
                starts[depth] += 1
                PC.append(copy.deepcopy(pair))
                vc_num = get_vc_num(param, pair)
                VC.append(vc_num)
                vc_len += vc_num
            starts[depth - 1] += 1
            depth -= 1
            continue
        if starts[depth] < len(param):
            pair[depth] = starts[depth]
            depth += 1
            starts[depth] = starts[depth - 1] + 1
            continue
        else:
            starts[depth - 1] += 1
            depth -= 1
    return PC,VC, vc_len

class CTEnv(gym.Env):
    mapIndex = []
    def __init__(self, name,t_way=2):
        self.cover_array = None
        self.tc_map = None
        self.name = name
        self.t_way = t_way
        self.param = InstanceParameter.prog_param[name]
        self.observation = None
        PC, VC,vc_len = get_all_tuple(self.param, t_way)
        self.PC = PC
        self.VC = VC
        self.size = np.prod(self.param)
        self.observation_space = spaces.MultiBinary(n = vc_len)
        self.action_space = spaces.Discrete(self.size)
        self.state = None
        self.observation = None
        self.vc_len = vc_len
        self.uncover = vc_len
        self.tc_set = []
        self.tran_list = InstanceParameter.get_TL(self.param)
        self.masks = None
        self.base = math.comb(len(self.param), self.t_way)
        if len(CTEnv.mapIndex) != self.size:
            CTEnv.mapIndex = [self.map_to_tc(i) for i in range(self.size)]

    def reset(self, seed=None):
        self.state = [[0]*i for i in self.VC]
        self.cover_array = [[0]*i for i in self.VC]
        self.uncover = self.vc_len
        self.tc_set = []
        action = np.random.randint(self.size)
        tc = self.map_to_tc(action)
        self.tc_set.append(tc)
        self.get_CM_reward(tc)
        self.observation = np.concatenate(self.state,dtype=np.double)
        self.masks = np.full(np.prod(self.param), True, dtype=bool)
        self.masks[action] = False
        return self.observation, self.get_info()

    def step(self, action):
        self.masks[action] = False
        tc = CTEnv.mapIndex[action]
        reward = self.get_CM_reward(tc)
        done = truncate = False
        self.tc_set.append(tc)
        if self.uncover <= 0:
            done = True
            reward += self.base
            self.nonrepeat()
        self.observation = np.concatenate(self.state)
        return self.observation, reward, done, truncate, self.get_info()

    def get_CM_reward(self, tc):
        reward = 0
        for idx, pc in enumerate(self.PC):
            vc_idx = tc[pc[0]] * self.param[pc[1]] + tc[pc[1]]
            self.cover_array[idx][vc_idx] += 1
            if self.state[idx][vc_idx] == 0:
                self.state[idx][vc_idx] = 1
                reward += 1
                self.uncover -= 1
        return reward

    def close(self):
        print("closing")

    def get_info(self):
        return { "ts_len": len(self.tc_set)}

    def seed(self, s):
        raise NotImplementedError

    def render(self, mode):
        raise NotImplementedError

    def get_tc_set(self):
        return self.tc_set

    def action_masks(self):
        return self.masks

    def map_to_tc(self, map_index):
        tc = []
        for ele in self.tran_list:
            res = divmod(map_index,ele)
            tc += [res[0]]
            map_index = res[1]
        return tc

    def nonrepeat(self):
        rem_list = []
        for i,tc in enumerate(self.tc_set):
            inx = []
            flag = True
            for idx, pc in enumerate(self.PC):
                count = tc[pc[-1]]
                pord = 1
                for i in range(len(pc) - 2, -1, -1):
                    pord *= self.param[pc[i + 1]]
                    count += tc[pc[i]] * pord
                inx.append(count)
                if self.cover_array[idx][count] <= 1:
                    flag = False
                    break
            if flag:
                for idx, pc in enumerate(self.PC):
                    self.cover_array[idx][inx[idx]] -= 1
                rem_list.append(i)
        if rem_list is not None and len(rem_list) > 0:
            for ele in rem_list[::-1]:
                self.tc_set.pop(ele)
