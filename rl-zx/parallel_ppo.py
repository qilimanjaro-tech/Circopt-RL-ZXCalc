import argparse
import os
import random
import time

from multiprocessing import Pool
import gymnasium as gym
import gym_zx
import networkx as nx
import numpy as np
import pyzx as zx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data

from rl_agent import AgentGNN

count = 0
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


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
    parser.add_argument("--num-envs", type=int, default=8,
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

    env = gym.make(gym_id, qubits=qubits, depth=depth, env_id= idx)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    return env



def step_env(env_action_pair):
    env, action = env_action_pair
    return env.step(action)

def reset_env(env):
    return env.reset()

def process_graph_obs(item):
    policy_items, value_items = item[0], item[1]
    value_graph = Data(x=value_items[0].cpu(), edge_index=value_items[1].cpu(), edge_attr=value_items[2].cpu())
    policy_graph = Data(x=policy_items[0].cpu(), edge_index=policy_items[1].cpu(), edge_attr=policy_items[2].cpu(), y=policy_items[3].cpu())
    return value_graph, policy_graph

if __name__ == "__main__":
    mp.set_start_method('spawn') ##set multiprocessing spawn for CUDA multiprocessing
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    #Training size
    qubits = 5
    depth = 100
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, qubits, depth) for i in range(args.num_envs)]
    agent = AgentGNN(envs[0], device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)   
    global_step = 0
    start_time = time.time()
    
    
    with Pool(processes=args.num_envs) as pool:
        for _ in range(0,1000):
            results = pool.map_async(reset_env, envs)
        pool.close()
        pool.join()
        reset = results.get()#returns a vector size (args.num_nenvs, 2) with each vector has obs0, reset_info
    end=time.time()
    print(end-start_time)
    reset_info_list = [item[1] for item in reset]
    all_graph_obs = [graph_obs for reset_info in reset_info_list for graph_obs in reset_info["graph_obs"]]
    """
    #Process the graph_obs data
    with Pool(processes=args.num_envs) as pool:
        processed_data = pool.map(process_graph_obs, all_graph_obs)

    # Unpack the processed data
    new_value_data, new_policy_data = zip(*processed_data)

    next_obs_graph = (Batch.from_data_list(new_policy_data), Batch.from_data_list(new_value_data))
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    cumulative_reward = []
    cumulative_episode_length = []
    action_counter = []
    action_nodes = []
    remaining_pivot_size = []
    remaining_lcomp_size = []
    cumulative_max_reward_difference = []
    action_patterns = []
    optimal_episode_length = []
    pyzx_gates = []
    rl_gates = []
    swap_gates = []
    pyzx_swap_gates = []
    wins_vs_pyzx = []
    """