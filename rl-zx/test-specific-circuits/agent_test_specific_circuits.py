import argparse
import json
import random
import time
from distutils.util import strtobool
import os 

import gymnasium as gym
import gym_zx
import pandas as pd
import numpy as np
import pyzx as zx
import torch
from torch_geometric.data import Batch, Data
from rl_agent import AgentGNN

global device

device = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments") 
    parser.add_argument("--num-episodes", type=int, default=1,
        help="the number of episodes to run")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gym-id", type=str, default="zx-v0",
        help="the id of the gym environment")

    return parser.parse_known_args()[0]


def make_env(gym_id, seed, idx, capture_video, run_name, qubits, gates, circuit):
    def thunk():
        env = gym.make(gym_id, qubits = qubits, depth = gates, circuit=circuit)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def get_results(param, circuit_data=None):
    rl_time, full_reduce_time = [], []
    args = parse_args()
    qubits, depth = param

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    run_name = "HP5"
    capture_video = False
    circuit = circuit_data[0]
    circuit_name = circuit_data[1]
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, capture_video, run_name, qubits, depth, circuit=circuit) for i in range(args.num_envs)]
    )

    agent = AgentGNN(envs, device).to(device)  
    path = os.getcwd()
    file_path = "/home/jan.nogue/radagast/home_content_jnogue/qilimanjaro/Circopt-RL-ZXCalc/rl-zx/test-specific-circuits/state_dict_195300005q_70_gflow_step_cflow_end_c2_init.pt"
    #file_path = os.path.join(path, "state_dict_model5x60_new.pt")
    agent.load_state_dict(
        torch.load(file_path, map_location=torch.device("cpu"))
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
        "twoqubit": [],
        "min_gates": [],
        "opt_episode_len": [],
        "episode_len": [],
        "opt_episode_len": [],
        "initial_2q": [],
        "action_stats": [],
        "depth": [],
        "initial_depth": [],
    }
    pyzx_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubit": []}
    bo_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubit": []}
    initial_stats = {
        "gates": [],
        "tcount": [],
        "clifford": [],
        "CNOT": [],
        "CX": [],
        "CZ": [],
        "had": [],
        "twoqubit": [],
    }
    rl_action_pattern = pd.DataFrame()
    final_circuit_data = circuit.stats_dict() #intial value of returning circuit
    for episode in range(args.num_episodes):  
        
        print(episode)
        done = False
        obs0, reset_info = envs.reset()
        new_value_data = []
        new_policy_data = []
        
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

            action, action_id = agent.get_action(state, device=device, deterministic=True)
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

        #rl_time.append(end - start)
        info = info["final_info"][0]
        rl_circ_s = info["rl_stats"]
        no_opt_s = info["no_opt_stats"]
        zx_circ_s = info["pyzx_stats"]
        in_circ_s = info["initial_stats"]

        #check if the final circuit has better stats than the previous episode, if it does keep it. Both in single and two qubit gates. 
        rl_zx_circuit = info["final_circuit"]
           
        if rl_circ_s["twoqubit"] == final_circuit_data["twoqubit"]:
            #check single qubits
            n0 = rl_circ_s["gates"] - rl_circ_s["twoqubit"]
            n1 = final_circuit_data["gates"] - final_circuit_data["twoqubit"]
            if n0 < n1:
                final_circuit_data = rl_zx_circuit.stats_dict()
                final_circuit = rl_zx_circuit
        elif rl_circ_s["twoqubit"] < final_circuit_data["twoqubit"]:
            final_circuit = rl_zx_circuit
            final_circuit_data = rl_zx_circuit.stats_dict()
                
            
        

        rl_stats["gates"].append(rl_circ_s["gates"])

        rl_stats["tcount"].append(rl_circ_s["tcount"])
        rl_stats["clifford"].append(rl_circ_s["clifford"])
        rl_stats["CNOT"].append(rl_circ_s["CNOT"])
        rl_stats["CZ"].append(rl_circ_s["CZ"])
        rl_stats["had"].append(rl_circ_s["had"])
        rl_stats["twoqubit"].append(rl_circ_s["twoqubit"])
        rl_stats["initial_2q"].append(no_opt_s["twoqubit"])
        rl_stats["episode_len"].append(info["episode_len"])
        rl_stats["opt_episode_len"].append(info["opt_episode_len"] + info["episode_len"])
        rl_stats["action_stats"].append(info["action_stats"])
        rl_stats["initial_depth"].append(info["initial_depth"])
        rl_stats["depth"].append(info["depth"])
        action_pattern_df = pd.DataFrame(info["action_pattern"])
        
        action_pattern_df["Episode"] = [episode]*action_pattern_df.shape[0]
        rl_action_pattern = pd.concat([rl_action_pattern, action_pattern_df], ignore_index=True)

        """ bo_stats["gates"].append(bo_circ_s["gates"])

        bo_stats["tcount"].append(bo_circ_s["tcount"])
        bo_stats["clifford"].append(bo_circ_s["clifford"])
        bo_stats["CNOT"].append(bo_circ_s["CNOT"])
        bo_stats["CZ"].append(bo_circ_s["CZ"])
        bo_stats["had"].append(bo_circ_s["had"])
        bo_stats["twoqubits"].append(bo_circ_s["twoqubits"])"""

        pyzx_stats["gates"].append(zx_circ_s["gates"])
        pyzx_stats["tcount"].append(zx_circ_s["tcount"])
        pyzx_stats["clifford"].append(zx_circ_s["clifford"])
        pyzx_stats["CNOT"].append(zx_circ_s["CNOT"])
        pyzx_stats["CZ"].append(zx_circ_s["CZ"])
        pyzx_stats["had"].append(zx_circ_s["had"])
        pyzx_stats["twoqubit"].append(zx_circ_s["twoqubit"])

        initial_stats["gates"].append(in_circ_s["gates"])
        initial_stats["tcount"].append(in_circ_s["tcount"])
        initial_stats["clifford"].append(in_circ_s["clifford"])
        initial_stats["CNOT"].append(in_circ_s["CNOT"])
        initial_stats["CZ"].append(in_circ_s["CZ"])
        initial_stats["had"].append(in_circ_s["had"])
        initial_stats["twoqubit"].append(in_circ_s["twoqubit"])


     

        wins += info["win_vs_pyzx"]

        print("Gates with RL", sum(rl_stats["gates"]) / len(rl_stats["gates"]))
        print("Gates with PyZX", sum(pyzx_stats["gates"]) / len(pyzx_stats["gates"]))
        #print("Gates with BOpt", sum(bo_stats["gates"]) / len(bo_stats["gates"]))
        print("2q with RL", sum(rl_stats["twoqubit"]) / len(rl_stats["twoqubit"]))
        print("2q with PyZX", sum(pyzx_stats["twoqubit"]) / len(pyzx_stats["twoqubit"]))
        #print("2q with BOpt", sum(bo_stats["twoqubits"]) / len(bo_stats["twoqubits"]))
        print("2q initial", sum(rl_stats["initial_2q"]) / len(rl_stats["initial_2q"]))
        print("Wins:", wins)

    #rl_action_pattern.to_csv("./results/specific_circuits/rl_action_pattern_"+str(qubits)+"x"+str(depth)+"_nc.json", index=False)  
   
    qasm_string = final_circuit.to_qasm()
    directory = "./results/specific_circuits_data/deterministic"+str(circuit_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, str(circuit_name)+"_opt") # Write the QASM content to the file 
    with open(file_path, 'w') as file: 
        file.write(qasm_string) 

    print(f'QASM file saved to: {file_path}')
    with open("./results/specific_circuits_data/deterministic∫"+str(circuit_name)+"/"+str(circuit_name)+"_rl_stats.json", "w") as f:
        json.dump(rl_stats, f)
    """ with open("./results/specific_circuits/"+str(circuit_name)+"/"+str(circuit_name)+"_pyzx_stats.json", "w") as f:
        json.dump(pyzx_stats, f)"""

    """with open("./results/specific_circuits/"+str(qubits)+"x"+str(depth)+"_nc.json", "w") as f:
        json.dump(initial_stats, f)
    with open("./results/specific_circuits/"+str(qubits)+"x"+str(depth)+"_nc.json", "w") as f:
        json.dump(bo_stats, f)"""
    
    return np.mean(full_reduce_time), np.mean(rl_time), np.std(full_reduce_time), np.std(rl_time)


import json
import multiprocessing as mp


if __name__ == "__main__": 
    folder_path = "/home/jan.nogue/radagast/home_content_jnogue/qilimanjaro/Circopt-RL-ZXCalc/rl-zx/test-specific-circuits/specific_circuits_data/deterministic"
    for filename in os.listdir(folder_path): 
        file_path = os.path.join(folder_path, filename) 
        if os.path.isfile(file_path):
            circuit = zx.circuit.Circuit.load(file_path).to_basic_gates()
            circuit_name = os.path.basename(file_path)

        fr_time_depth, rl_time_depth = [],[]
        fr_time_var, rl_time_var = [],[]

        qubit = circuit.qubits
        depth = circuit.depth()
        fr_time, rl_time, fr_var, rl_var = get_results((qubit,depth), circuit_data=(circuit,circuit_name))
        
            
