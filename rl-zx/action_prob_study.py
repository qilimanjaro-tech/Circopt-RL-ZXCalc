from rl_agent import AgentGNN
import gymnasium as gym
import gym_zx
import torch
from torch_geometric.data import Batch, Data
import pyzx as zx
import torch.multiprocessing as mp
import argparse
import os
import json
import numpy as np


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="zx-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=8983440,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments") #default 8
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.05,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def make_env(gym_id, seed, idx, capture_video, run_name, qubits, depth):
    
    def thunk():
        env = gym.make(gym_id, qubits=qubits, depth=depth)
        return env

    return thunk


if __name__ == "__main__":
    mp.set_start_method('spawn') ##set multiprocessing spawn for CUDA multiprocessing
    args = parse_args()
    #SrH_env = gym.vector.AsyncVectorEnv(
    #   [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name="test", qubits=12, depth=70) for i in range(args.num_envs)], shared_memory=False)
    gates_list = [60,520,1000,1560,2100]
    qubits_list = [5,10,20,40,80]
    for qubits,gates in zip(qubits_list, gates_list):
        stats = {
            "action_dist" : [],
            "episode_len": [],
            "opt_episode_len": [],
            "diff_episode_len":[],
            "stop_evolution":[],
            "len_actions_evolution": [],
            "max_prob": [],
            "remaining_lcomp":[],
            "remaining_pivot":[],
            "remaining_id":[]

        }
        SrH_env = gym.make('zx-v0', qubits=qubits, depth=gates)
        """SrH_env = gym.vector.AsyncVectorEnv(
            [make_env("zx-v0", args.seed + i, i, args.capture_video, run_name="test", qubits=qubits, depth=gates) for i in range(args.num_envs)], shared_memory=False)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rl_agent = AgentGNN(None,device=device).to(device)
        rl_agent.load_state_dict(
                torch.load("/home/jnogue/qilimanjaro/Circopt-RL-ZXCalc/rl-zx/state_dict_model5x60_new.pt", map_location=torch.device("cpu"))
            )  
        rl_agent.eval()
    
        for i in range(100):
            print(f"Qubits: {qubits}, Gates: {gates}, Iteration: {i}")
            probs_list = [] #list of probabilities for maximum episode len 
            obs0, reset_info = SrH_env.reset()
           
            policy_items, value_items = reset_info["graph_obs"]
            value_graph = [Data(x=value_items[0], edge_index=value_items[1], edge_attr=value_items[2])]
            policy_graph = [Data(x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3])]
            next_obs_graph = (Batch.from_data_list(policy_graph), Batch.from_data_list(value_graph))
            done = False
            while not done:
                
                action, logprob, _, value, logits, action_ids, probs = rl_agent.get_action_and_value(next_obs_graph, device=device)
                probs_list.append(probs.cpu().squeeze().tolist())
                next_obs, reward, done, deprecated, info = SrH_env.step(action_ids.cpu().numpy())
               
                policy_items, value_items = info["graph_obs"]
                value_graph = [Data(x=value_items[0], edge_index=value_items[1], edge_attr=value_items[2])]
                policy_graph = [Data(x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3])]
                next_obs_graph = (Batch.from_data_list(policy_graph), Batch.from_data_list(value_graph))
            stop_evolution, len_action_evolution, max_prob = [], [], []
            for probs in probs_list[info["single_opt_episode_len"]:]:
                stop_evolution.append(probs[-1])
                len_action_evolution.append(len(probs))
                max_prob.append(np.max(probs))
            
            stats["remaining_lcomp"].append(info["remaining_lcomp_size"])
            stats["remaining_pivot"].append(info["remaining_pivot_size"])
            stats["remaining_id"].append(info["remaining_id_size"])
            stats["max_prob"].append(max_prob)
            stats["len_actions_evolution"].append(len_action_evolution)
            stats["stop_evolution"].append(stop_evolution)
            stats["action_dist"].append(probs_list[info["single_opt_episode_len"]])
            stats["episode_len"].append(info["episode_len"])
            stats["opt_episode_len"].append(info["single_opt_episode_len"]) #optimal episode_len
            stats["diff_episode_len"].append(info["opt_episode_len"]) #this is opt_len - episode_len

        filename = './results/5x60_non_clifford/data/action_dist/opt_episode_len_probs'+str(qubits)+'x'+str(gates)+'.json'
    
        with open(filename, 'w') as json_file:
            json .dump(stats, json_file)    
        print(f"{i+1} file generated")
            
            

