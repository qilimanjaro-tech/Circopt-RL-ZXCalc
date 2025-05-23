import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import os

from torch.distributions.categorical import Categorical
from torch_geometric.nn import Sequential as geo_Sequential
from torch_geometric.nn.aggr import AttentionalAggregation

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, device="cuda"):
        if masks is None:
            masks = []
        self.masks = masks
        if len(self.masks) != 0:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self, device):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class AgentGNN(nn.Module):
    def __init__(
        self,
        envs,
        device,
        c_hidden=128,
        c_hidden_v=128,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        #obs_shape = envs.get_attr("shape")#3000
        #self.obs_shape = obs_shape[0]
        #qubits_list = envs.get_attr('qubits')#retrieve environemnt qubits
        #self.qubits = qubits_list[0]
        self.obs_shape = 3000
        self.bin_required = int(np.ceil(np.log2(self.obs_shape)))

        c_in_p = 17
        c_in_v = 12
        edge_dim = 7
        edge_dim_v = 2
        self.global_attention_critic = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(c_hidden, c_hidden),
                nn.ReLU(),
                nn.Linear(c_hidden, c_hidden),
                nn.ReLU(),
                nn.Linear(c_hidden, 1),
            ),
            nn=nn.Sequential(nn.Linear(c_hidden, c_hidden_v), nn.ReLU(), nn.Linear(c_hidden_v, c_hidden_v), nn.ReLU()),
        )

        self.critic_gnn = geo_Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    geom_nn.GATv2Conv(c_in_v, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
            ],
        )

        self.actor_gnn = geom_nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    geom_nn.GATv2Conv(c_in_p, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (nn.Linear(c_hidden, c_hidden),),
                nn.ReLU(),
                (nn.Linear(c_hidden, 1),),
            ],
        )

        self.critic_ff = nn.Sequential(
            nn.Linear(c_hidden_v, c_hidden_v),
            nn.ReLU(),
            nn.Linear(c_hidden_v, c_hidden_v),
            nn.ReLU(),
            nn.Linear(c_hidden_v, out_features=1),
        )

    def actor(self, x):
        logits = self.actor_gnn(x.x, x.edge_index, x.edge_attr)
        return logits

    def critic(self, x):
        features = self.critic_gnn(x.x, x.edge_index, x.edge_attr)
        aggregated = self.global_attention_critic(features, x.batch)
        return self.critic_ff(aggregated)
    
    def get_action(self, x, device="cuda"):
        policy_obs, _ = x
        logits = self.actor(policy_obs)
        
        batch_logits = torch.zeros([x[0].num_graphs, self.obs_shape]).to(device)
        act_mask = torch.zeros([x[0].num_graphs, self.obs_shape]).to(device)
        act_ids = torch.zeros([x[0].num_graphs, self.obs_shape]).to(device)
        action_logits = torch.tensor([]).to(device)

        for b in range(x[0].num_graphs):
            
            ids = x[0].y[x[0].batch == b].to(device)
            action_nodes = torch.where(ids != -1)[0].to(device)
            probs = logits[x[0].batch == b][action_nodes]
            batch_logits[b, : probs.shape[0]] = probs.flatten()
            act_mask[b, : probs.shape[0]] = torch.tensor([True] * probs.shape[0])
            act_ids[b, : action_nodes.shape[0]] = ids[action_nodes]
            action_logits = torch.cat((action_logits, probs.flatten()), 0).reshape(-1)
            
        # Sample from each set of probs using Categorical
        categoricals = CategoricalMasked(logits=batch_logits, masks=act_mask, device=device)

        # Convert the list of samples back to a tensor
        action = categoricals.sample()
        #action = torch.randint(low=0, high=probs.shape[0], size=(1,))
        #action = torch.argmax(probs, dim=0)
        batch_id = torch.arange(x[0].num_graphs)
        action_id = act_ids[batch_id, action]

        return action.permute(*torch.arange(action.ndim - 1, -1, -1)), action_id.permute(*torch.arange(action_id.ndim - 1, -1, -1))

    def get_action_and_value(self, x, action=None, device="cpu", testing=False):
        
        policy_obs, value_obs = x
        logits = self.actor(policy_obs)
        values = self.critic(value_obs)
        
        batch_logits = torch.zeros([x[0].num_graphs, self.obs_shape]).to(device)
        act_mask = torch.zeros([x[0].num_graphs, self.obs_shape]).to(device)
        act_ids = torch.zeros([x[0].num_graphs, self.obs_shape]).to(device)
        action_logits = torch.tensor([]).to(device)

        for b in range(x[0].num_graphs):
            
            ids = x[0].y[x[0].batch == b].to(device)
            action_nodes = torch.where(ids != -1)[0].to(device)
            probs = logits[x[0].batch == b][action_nodes]
            batch_logits[b, : probs.shape[0]] = probs.flatten()
            act_mask[b, : probs.shape[0]] = torch.tensor([True] * probs.shape[0])
            act_ids[b, : action_nodes.shape[0]] = ids[action_nodes]
            action_logits = torch.cat((action_logits, probs.flatten()), 0).reshape(-1)
            
        # Sample from each set of probs using Categorical
        categoricals = CategoricalMasked(logits=batch_logits, masks=act_mask, device=device)

        # Convert the list of samples back to a tensor
        values = values.squeeze(-1)
        if action is None:
            action = categoricals.sample()
            batch_id = torch.arange(x[0].num_graphs)
            action_id = act_ids[batch_id, action]

        else:
            action_id = torch.tensor([0]).to(device)
            
        if testing:
            return action.permute(*torch.arange(action.ndim - 1, -1, -1)), action_id.permute(*torch.arange(action_id.ndim - 1, -1, -1))
        
        logprob = categoricals.log_prob(action)
        entropy = categoricals.entropy(device)
        return action.permute(*torch.arange(action.ndim - 1, -1, -1)), logprob, entropy, values, action_logits.clone().detach().to(device).reshape(-1, 1), action_id.permute(*torch.arange(action_id.ndim - 1, -1, -1))
        

    def get_value(self, x):
        values = self.critic(x)
        return values