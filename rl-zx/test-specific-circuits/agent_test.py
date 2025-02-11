import argparse
import json
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import gym_zx
import pandas as pd
import numpy as np
import pyzx as zx
import torch
from torch_geometric.data import Batch, Data
from rl_agent import AgentGNN

global device

device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments") 
    parser.add_argument("--num-episodes", type=int, default=1000,
        help="the number of episodes to run")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--seed", type=int, default=10000,
        help="seed of the experiment")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gym-id", type=str, default="zx-v0",
        help="the id of the gym environment")

    return parser.parse_known_args()[0]


def make_env(gym_id, seed, idx, capture_video, run_name, qubits, depth, toffoli,circuit, basic_opt,tele_reduce):
    
    def thunk():
        env = gym.make(gym_id, qubits=qubits, depth=depth, env_id= idx,circuit=circuit, toffoli=toffoli, basic_opt=basic_opt, tele_red=tele_reduce, QAOA=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk

def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]  # Apply recursively to each item in list
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}  # Apply recursively to each item in dict
    else:
        return obj  # 

def get_results(param):
    rl_time, full_reduce_time = [], []
    args = parse_args()
    qubits, depth = param

    toffoli=False
    #circuit= zx.Circuit.from_qasm_file(dirname + circuit_name).to_basic_gates()
    circuit=None
    basic_opt=False
    tele_reduce=False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    run_name = "HP5"
    capture_video = False
    envs = gym.vector.SyncVectorEnv(
       [make_env(args.gym_id, args.seed + i, i, False, run_name, qubits, depth, toffoli, circuit, basic_opt, tele_reduce) for i in range(args.num_envs)])

    agent = AgentGNN(envs, device).to(device)  

    agent.load_state_dict(
        torch.load("/Users/jordi/Research/Copt-cquere/rl-zx/state_dict_50400005q_70_no_basic_opt.pt", map_location=torch.device("cpu"))
    )  
    agent.eval()
   
    done = False
    wins = 0
    rl_stats = {
        "gates": [],
        "tcount": [],
        "clifford": [],
        "CNOT": [],
        "CX": [],
        "CZ": [],
        "had": [],
        "twoqubits": [],
        "min_gates": [],
        "opt_episode_len": [],
        "episode_len": [],
        "opt_episode_len": [],
        "initial_2q": [],
        "action_stats": [],
        "depth": [],
        "initial_depth": [],
    }
    pyzx_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubits": []}
    #bo_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubits": []}
    initial_stats = {
        "gates": [],
        "tcount": [],
        "clifford": [],
        "CNOT": [],
        "CX": [],
        "CZ": [],
        "had": [],
        "twoqubits": [],
    }
    rl_action_pattern = pd.DataFrame()
    #actions_available = [0,0,0,0]
    for episode in range(args.num_episodes):  
        
        print(episode)
        min_gates = np.inf
        obs0, reset_info = envs.reset()
        #actions_available = [actions_available[i] + reset_info["actions_available"][0][i] for i in range(4)]
        done = False
        
        new_value_data = []
        new_policy_data = []
        #full_reduce_time.append(reset_info["full_reduce_time"])
        
        for item in reset_info["graph_obs"]:
            policy_items, value_items = item[0], item[1]
            value_graph = Data(x=value_items[0], edge_index=value_items[1])
            policy_graph = Data(
                x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3]
            )
            new_value_data.append(value_graph)
            new_policy_data.append(policy_graph)

        next_obs_graph = (
            Batch.from_data_list(new_policy_data).to(device),
            Batch.from_data_list(new_value_data).to(device),
        )
        state = next_obs_graph
        start = time.time()
        while not done:

            action, action_id = agent.get_action(state, device=device)
            action = action.flatten()
            
            next_obs, reward, done, deprecated, info = envs.step(action_id.cpu().numpy())
            new_value_data = []
            new_policy_data = []

            for item in info["graph_obs"]:
                policy_items, value_items = item[0], item[1]
                value_graph = Data(x=value_items[0], edge_index=value_items[1])
                policy_graph = Data(
                    x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3]
                )
                new_value_data.append(value_graph)
                new_policy_data.append(policy_graph)

            next_obs_graph = (
                Batch.from_data_list(new_policy_data).to(device),
                Batch.from_data_list(new_value_data).to(device),
            )
            next_done = torch.zeros(args.num_envs).to(device)
            state = next_obs_graph
        end = time.time()
    

        rl_time.append(end - start)
        #info = info["final_info"][0]
        rl_circ_s = info["rl_stats"]
        #bo_circ_s = info["bo_stats"]
        no_opt_s = info["no_opt_stats"]
        zx_circ_s = info["pyzx_stats"]
        in_circ_s = info["initial_stats"]

        rl_stats["gates"].append(rl_circ_s["gates"][0])

        rl_stats["tcount"].append(rl_circ_s["tcount"][0])
        rl_stats["clifford"].append(rl_circ_s["clifford"][0])
        rl_stats["CNOT"].append(rl_circ_s["CNOT"][0])
        rl_stats["CZ"].append(rl_circ_s["CZ"][0])
        rl_stats["had"].append(rl_circ_s["had"][0])
        rl_stats["twoqubits"].append(rl_circ_s["twoqubits"][0])
        rl_stats["initial_2q"].append(no_opt_s["twoqubits"][0])
        rl_stats["episode_len"].append(info["episode_len"][0])
        rl_stats["opt_episode_len"].append(info["opt_episode_len"][0] + info["episode_len"][0])
        rl_stats["action_stats"].append(info["action_stats"][0])
        rl_stats["initial_depth"].append(info["initial_depth"][0])
        rl_stats["depth"].append(info["depth"][0])
        
        rl_stats=convert_numpy_types(rl_stats)
        action_pattern_df = pd.DataFrame(info["action_pattern"])
        
        action_pattern_df["Episode"] = [episode]*action_pattern_df.shape[0]
        rl_action_pattern = pd.concat([rl_action_pattern, action_pattern_df], ignore_index=True)

        """
        bo_stats["gates"].append(bo_circ_s["gates"][0])

        bo_stats["tcount"].append(bo_circ_s["tcount"][0])
        bo_stats["clifford"].append(bo_circ_s["clifford"][0])
        bo_stats["CNOT"].append(bo_circ_s["CNOT"][0])
        bo_stats["CZ"].append(bo_circ_s["CZ"][0])
        bo_stats["had"].append(bo_circ_s["had"][0])
        bo_stats["twoqubits"].append(bo_circ_s["twoqubits"][0])
        """
        pyzx_stats["gates"].append(zx_circ_s["gates"][0])
        pyzx_stats["tcount"].append(zx_circ_s["tcount"][0])
        pyzx_stats["clifford"].append(zx_circ_s["clifford"][0])
        pyzx_stats["CNOT"].append(zx_circ_s["CNOT"][0])
        pyzx_stats["CZ"].append(zx_circ_s["CZ"][0])
        pyzx_stats["had"].append(zx_circ_s["had"][0])
        pyzx_stats["twoqubits"].append(zx_circ_s["twoqubits"][0])

        initial_stats["gates"].append(in_circ_s["gates"][0])
        initial_stats["tcount"].append(in_circ_s["tcount"][0])
        initial_stats["clifford"].append(in_circ_s["clifford"][0])
        initial_stats["CNOT"].append(in_circ_s["CNOT"][0])
        initial_stats["CZ"].append(in_circ_s["CZ"][0])
        initial_stats["had"].append(in_circ_s["had"][0])
        initial_stats["twoqubits"].append(in_circ_s["twoqubits"][0])

        #bo_stats=convert_numpy_types(bo_stats)
        initial_stats=convert_numpy_types(initial_stats)
        pyzx_stats=convert_numpy_types(pyzx_stats)
        wins += (info["win_vs_pyzx"] if info["win_vs_pyzx"]==1 else 0)

        #print("Gates with RL", sum(rl_stats["gates"]) / len(rl_stats["gates"]))
        #print("Gates with PyZX", sum(pyzx_stats["gates"]) / len(pyzx_stats["gates"]))
        #print("Gates with BOpt", sum(bo_stats["gates"]) / len(bo_stats["gates"]))
        #print("2q with RL", sum(rl_stats["twoqubits"]) / len(rl_stats["twoqubits"]))
        #print("2q with BOpt", sum(bo_stats["twoqubits"]) / len(bo_stats["twoqubits"]))
        #print("2q with PyZX", sum(pyzx_stats["twoqubits"]) / len(pyzx_stats["twoqubits"]))
        #print("2q initial", sum(rl_stats["initial_2q"]) / len(rl_stats["initial_2q"]))
        print("Wins:", wins*1.0/len(rl_stats["initial_2q"]))
        #print("Actions Available: ", [actions_available[i]/len(rl_stats["initial_2q"]) for i in range(4)])

    #rl_action_pattern.to_csv("./results/qaoa_4qubits/rl_action_pattern_"+str(qubits)+"x"+str(depth)+"_nc.json", index=False)  
   
    with open("/Users/jordi/Research/Copt-cquere/rl-zx/results/qaoa_4qubits/rl_stats_stopping_"+str(qubits)+"x"+str(depth)+"_nc.json", "w") as f:
        json.dump(rl_stats, f)
    with open("/Users/jordi/Research/Copt-cquere/rl-zx/results/qaoa_4qubits/pyzx_stats_stopping_"+str(qubits)+"x"+str(depth)+"_nc.json", "w") as f:
        json.dump(pyzx_stats, f)

    with open("/Users/jordi/Research/Copt-cquere/rl-zx/results/qaoa_4qubits/initial_stats_stopping_"+str(qubits)+"x"+str(depth)+"_nc.json", "w") as f:
        json.dump(initial_stats, f)
    
    #with open("/Users/jordi/Research/Copt-cquere/rl-zx/results/qaoa_4qubits/bo_stats_stopping_"+str(qubits)+"x"+str(depth)+"_nc.json", "w") as f:
    #    json.dump(bo_stats, f)
    
    return 


import json
import multiprocessing as mp

if __name__ == "__main__":
    qubits = [10]
    depths = [210]
    for qubit in qubits:
        fr_time_depth, rl_time_depth = [],[]
        fr_time_var, rl_time_var = [],[]
        for depth in depths:
            print(f"Qubits, Depth {qubit, depth}")
            get_results((qubit,depth))
            
            
        #with open("./results/qaoa_4qubits/time_depth_"+str(qubit)+ ".json", "w") as f:
        #    json.dump({"full_time":fr_time_depth, "rl_time":rl_time_depth, "full_var":fr_time_var, "rl_var":rl_time_var}, f)