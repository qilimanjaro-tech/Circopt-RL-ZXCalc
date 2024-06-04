import argparse
import os
import random
import time


import gym
import gym_zx
import networkx as nx
import numpy as np
import pyzx as zx
import torch
import torch.nn as nn
import torch.optim as optim

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
    parser.add_argument("--num-envs", type=int, default=1,
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
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    #Training size
    qubits = 5
    depth = 55
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, qubits, depth) for i in range(args.num_envs)]
    )
    agent = AgentGNN(envs, device).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)    

    
    global_step = 0
    start_time = time.time()
    obs0, reset_info = envs.reset()
    new_value_data = []
    new_policy_data = []
    for item in reset_info["graph_obs"]:
        policy_items, value_items = item[0], item[1]
        value_graph = Data(x=value_items[0], edge_index=value_items[1], edge_attr=value_items[2])
        policy_graph = Data(x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3])
        new_value_data.append(value_graph)
        new_policy_data.append(policy_graph)

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
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if update % 50 == 1:
            torch.save(agent.state_dict(), "state_dict_" + str(global_step) + "model5x70_twoqubits_new.pt")
        if args.anneal_lr:
            frac = max(1.0 / 100, 1.0 - (update - 1.0) / (num_updates * 5.0 / 6))
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            neg_reward_discount = max(1, 5 * (1 - 4 * update / num_updates))
        if update * 1.0 / num_updates > 5.0 / 6:
            ent_coef = 0
        else:
            ent_coef = args.ent_coef
        value_data = []
        policy_data = []
        for step in range(args.num_steps):  
            global_step += 1 * args.num_envs
            value_data.extend(new_value_data)
            policy_data.extend(new_policy_data)
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, logits, action_ids = agent.get_action_and_value(next_obs_graph, device=device)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, deprecated, info = envs.step(action_ids.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_done = torch.Tensor(done).to(device)

            new_value_data = []
            new_policy_data = []
            for item in info["graph_obs"]:
                policy_items, value_items = item[0], item[1]
                value_graph = Data(x=value_items[0], edge_index=value_items[1], edge_attr=value_items[2])
                policy_graph = Data(
                    x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3]
                )
                new_value_data.append(value_graph)
                new_policy_data.append(policy_graph)

            next_obs_graph = (Batch.from_data_list(new_policy_data), Batch.from_data_list(new_value_data))

            if "action" in info.keys():
                for element in info["action"]:
                    action_counter.append(element)

                for element in info["nodes"]:
                    if element is not None:
                        for node in element:
                            action_nodes.append(node)

            if info != {} and "final_info" in info.keys():
                for idx, item in enumerate(info["final_info"]):
                    if done[idx]:
                        # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        cumulative_reward.append(item["episode"]["r"])
                        cumulative_episode_length.append(item["episode"]["l"])
                        remaining_pivot_size.append(item["remaining_pivot_size"])
                        remaining_lcomp_size.append(item["remaining_lcomp_size"])
                        cumulative_max_reward_difference.append(item["max_reward_difference"])
                        action_patterns.append(item["action_pattern"])
                        action_counter.append(item["action"])
                        optimal_episode_length.append(item["opt_episode_len"])
                        pyzx_gates.append(item["pyzx_gates"])
                        rl_gates.append(item["rl_gates"])
                        swap_gates.append(item["swap_cost"])
                        pyzx_swap_gates.append(item["pyzx_swap_cost"])
                        wins_vs_pyzx.append(item["win_vs_pyzx"])

        # bootstrap value if not done, implement GAE-Lambda advantage calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs_graph[1]).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)  
        clipfracs = []
        for epoch in range(args.update_epochs):
            
            np.random.shuffle(b_inds)  

            for start in range(
                0, args.batch_size, args.minibatch_size
            ):  # loop over entire batch, one minibatch at the time
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                values_batch = Batch.from_data_list([value_data[i] for i in mb_inds])
                policies_batch = Batch.from_data_list([policy_data[i] for i in mb_inds])

                _, newlogprob, entropy, newvalue, logits, _ = agent.get_action_and_value(
                    (policies_batch, values_batch),
                    None,
                    b_actions.long()[mb_inds].T, device=device
                )  # training begins, here we pass minibatch action so the agent doesnt sample a new action
                logratio = newlogprob - b_logprobs[mb_inds]  # logratio = log(newprob/oldprob)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/mean_reward", sum(cumulative_reward) / len(cumulative_reward), global_step)
        writer.add_scalar(
            "charts/mean_episode_length",
            sum(cumulative_episode_length) * 1.0 / len(cumulative_episode_length),
            global_step,
        )
        writer.add_scalar(
            "charts/remaining_pivot_size_mean", sum(remaining_pivot_size) * 1.0 / len(remaining_pivot_size), global_step
        )
        writer.add_scalar(
            "charts/remaining_lcomp_size_mean", sum(remaining_lcomp_size) * 1.0 / len(remaining_lcomp_size), global_step
        )
        writer.add_scalar(
            "charts/max_reward_difference_mean",
            sum(cumulative_max_reward_difference) / len(cumulative_max_reward_difference),
            global_step,
        )
        writer.add_scalar(
            "charts/opt_episode_len_mean", sum(optimal_episode_length) / len(optimal_episode_length), global_step
        )
        writer.add_scalar("charts/pyzx_gates", sum(pyzx_gates) / len(pyzx_gates), global_step)
        writer.add_scalar("charts/rl_gates", sum(rl_gates) / len(rl_gates), global_step)
        writer.add_scalar("charts/wins_vs_pyzx", sum(wins_vs_pyzx) / len(wins_vs_pyzx), global_step)
        writer.add_scalar("charts/value_function", torch.mean(b_values), global_step)
        writer.add_scalar("charts/swap_gates", sum(swap_gates) / len(swap_gates), global_step)
        writer.add_scalar("charts/pyzx_swap_gates", sum(pyzx_swap_gates) / len(pyzx_swap_gates), global_step)

        print(
            "rl_gates: ",
            sum(rl_gates) / len(rl_gates),
            " pyzx_gates: ",
            sum(pyzx_gates) / len(pyzx_gates),
            " wins: ",
            sum(wins_vs_pyzx) / len(wins_vs_pyzx),
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_histogram("histograms/reward_distribution", np.array(cumulative_reward), global_step)
        writer.add_histogram("histograms/episode_length_distribution", np.array(cumulative_episode_length), global_step)
        writer.add_histogram("histograms/action_counter_distribution", np.array(action_counter), global_step)
        writer.add_histogram("histograms/action_nodes_distribution", np.array(action_nodes), global_step)
        writer.add_histogram(
            "histograms/remaining_pivot_size_distribution", np.array(remaining_pivot_size), global_step
        )
        writer.add_histogram(
            "histograms/remaining_lcomp_size_distribution", np.array(remaining_lcomp_size), global_step
        )
        writer.add_histogram(
            "histograms/max_reward_difference_distribution", np.array(cumulative_max_reward_difference), global_step
        )
        writer.add_histogram("histograms/opt_episode_len_distribution", np.array(optimal_episode_length), global_step)
        writer.add_histogram("histograms/rl_gates", np.array(rl_gates), global_step)
        writer.add_histogram("histograms/pyzx_gates", np.array(pyzx_gates), global_step)
        writer.add_histogram(
            "histograms/value_function",
            b_values.cpu()
            .detach()
            .numpy()
            .reshape(
                -1,
            ),
            global_step,
        )
        writer.add_histogram(
            "histograms/logits",
            logits.cpu()
            .detach()
            .numpy()
            .reshape(
                -1,
            ),
            global_step,
        )

        cumulative_episode_length = []
        cumulative_reward = []
        action_counter = []
        action_nodes = []
        action_mask_size = []
        remaining_lcomp_size = []
        remaining_pivot_size = []
        cumulative_max_reward_difference = []
        action_patterns = []
        optimal_episode_length = []
        pyzx_gates = []
        rl_gates = []
        swap_gates = []
        pyzx_swap_gates = []
        wins_vs_pyzx = []
    envs.close()
    writer.close()

torch.save(agent.state_dict(), "state_dict_model5x70_twoqubits_new.pt")