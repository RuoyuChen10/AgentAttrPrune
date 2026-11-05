import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio
import math

from SafeSieve.graph.node import Node
from SafeSieve.agents.agent_registry import AgentRegistry
from .pruning_algorithm import ZeroExtensionPruner

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str] = None,
                agent_names: List[str] = None,
                decision_method: str = None,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                # New parameter for heterogeneous LLM support
                heterogeneous_llms: bool = False,
                llm_config: List[str] = None,
                ):
        
        # Heterogeneous LLM configuration
        if heterogeneous_llms and llm_config:
            # Validate llm_config
            if not isinstance(llm_config, list) or len(llm_config) == 0:
                raise ValueError("llm_config must be a non-empty list when heterogeneous_llms is True")
            self.heterogeneous_llms = True
            self.llm_config = llm_config
            # Assign an LLM to each agent, loop if not enough
            self.agent_llm_assignments = []
            for i in range(len(agent_names)):
                self.agent_llm_assignments.append(llm_config[i % len(llm_config)])
            # For backward compatibility, use the first LLM as the default value
            self.llm_name = llm_config[0] if llm_config else llm_name
        else:
            # Homogeneous LLM configuration (original logic)
            self.heterogeneous_llms = False
            self.llm_config = [llm_name] if llm_name else []
            self.agent_llm_assignments = [llm_name] * len(agent_names) if llm_name and agent_names else []
            self.llm_name = llm_name
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        # The decision node uses the first LLM or the specified LLM
        decision_llm = self.agent_llm_assignments[0] if self.agent_llm_assignments else self.llm_name
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":decision_llm})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        # Add mapping from node ID to index
        self.node_id_to_index = {}
        # record node order in __init__
        self.node_order = []  # Record node order
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
                                                 requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks

    def detect_isolated_nodes(self, min_agents: int = 2) -> List[str]:
        """
        Detect isolated nodes (nodes with all zeros in the mask)
        
        Args:
            min_agents: Minimum number of agents to keep
            
        Returns:
            List of node IDs to be discarded
        """
        if len(self.nodes) <= min_agents:
            return []
            
        isolated_nodes = []
        N = len(self.node_order)
        
        for i, node_id in enumerate(self.node_order):
            is_isolated = True
            
            # Check spatial connections
            if self.optimized_spatial:
                spatial_connected = False
                for j in range(N):
                    if i != j:  # Exclude self-loop
                        out_idx = i * N + j  # Outgoing edge
                        in_idx = j * N + i  # Incoming edge
                        if (out_idx < len(self.spatial_masks) and self.spatial_masks[out_idx].item() > 0) or \
                           (in_idx < len(self.spatial_masks) and self.spatial_masks[in_idx].item() > 0):
                            spatial_connected = True
                            break
                if spatial_connected:
                    is_isolated = False
            
            # Check temporal connections
            if self.optimized_temporal and is_isolated:
                temporal_connected = False
                for j in range(N):
                    if i != j:  # Exclude self-loop
                        out_idx = i * N + j  # Outgoing edge
                        in_idx = j * N + i  # Incoming edge
                        if (out_idx < len(self.temporal_masks) and self.temporal_masks[out_idx].item() > 0) or \
                           (in_idx < len(self.temporal_masks) and self.temporal_masks[in_idx].item() > 0):
                            temporal_connected = True
                            break
                if temporal_connected:
                    is_isolated = False
            
            # If neither spatial nor temporal is optimized, or the node is indeed isolated
            if (not self.optimized_spatial and not self.optimized_temporal) or is_isolated:
                isolated_nodes.append(node_id)
        
        # Ensure not to discard too many nodes so that the number does not fall below the minimum
        if len(self.nodes) - len(isolated_nodes) < min_agents:
            # Keep some nodes to meet the minimum requirement
            keep_count = len(self.nodes) - min_agents
            isolated_nodes = isolated_nodes[:keep_count]
            
        return isolated_nodes

    def remove_isolated_nodes(self, isolated_node_ids: List[str]) -> tuple:
        if not isolated_node_ids:
            return False, [], []
        print(f"Removing {len(isolated_node_ids)} isolated nodes: {isolated_node_ids}")
        # Record the order and masks before removal
        old_node_order = self.node_order.copy()
        old_id_to_index = {nid: idx for idx, nid in enumerate(old_node_order)}
        old_N = len(old_node_order)
        old_spatial_mask = self.spatial_masks.detach().cpu().numpy().reshape(old_N, old_N) if self.optimized_spatial else None
        old_temporal_mask = self.temporal_masks.detach().cpu().numpy().reshape(old_N, old_N) if self.optimized_temporal else None
        
        # record the removed node indices
        removed_node_indices = []
        for node_id in isolated_node_ids:
            if node_id in old_id_to_index:
                removed_node_indices.append(old_id_to_index[node_id])
        
        # Remove isolated nodes
        for node_id in isolated_node_ids:
            if node_id in self.nodes:
                del self.nodes[node_id]
            if node_id in self.node_order:
                self.node_order.remove(node_id)
        # Re-number the remaining nodes
        self.node_id_to_index = {nid: idx for idx, nid in enumerate(self.node_order)}
        # Re-build agent_names and node_kwargs based on the remaining nodes
        remaining_node_ids = list(self.nodes.keys())
        new_agent_names = []
        new_node_kwargs = []
        for i, agent_name in enumerate(self.agent_names):
            node_id = None
            for nid, idx in self.node_id_to_index.items():
                if idx == i and nid in remaining_node_ids:
                    node_id = nid
                    break
            if node_id is not None:
                new_agent_names.append(agent_name)
                new_node_kwargs.append(self.node_kwargs[i])
        self.agent_names = new_agent_names
        self.node_kwargs = new_node_kwargs
        # Re-initialize potential edges
        self.potential_spatial_edges = []
        self.potential_temporal_edges = []
        self.init_potential_edges()
        # Re-initialize logits and masks, preserving the original sparsity
        N = len(self.node_order)
        if self.optimized_spatial:
            # New spatial mask
            new_spatial_mask = np.zeros((N, N), dtype=np.float32)
            for i, nid_i in enumerate(self.node_order):
                for j, nid_j in enumerate(self.node_order):
                    if nid_i in old_id_to_index and nid_j in old_id_to_index:
                        old_i, old_j = old_id_to_index[nid_i], old_id_to_index[nid_j]
                        new_spatial_mask[i, j] = old_spatial_mask[old_i, old_j]
            self.spatial_masks = torch.nn.Parameter(torch.tensor(new_spatial_mask.flatten()), requires_grad=False)
            # Reset logits to 0 (optional, or keep the original)
            init_spatial_logit = torch.log(torch.tensor(0.5 / 0.5))
            self.spatial_logits = torch.nn.Parameter(
                torch.ones(N*N, requires_grad=True) * init_spatial_logit,
                requires_grad=True
            )
        if self.optimized_temporal:
            new_temporal_mask = np.zeros((N, N), dtype=np.float32)
            for i, nid_i in enumerate(self.node_order):
                for j, nid_j in enumerate(self.node_order):
                    if nid_i in old_id_to_index and nid_j in old_id_to_index:
                        old_i, old_j = old_id_to_index[nid_i], old_id_to_index[nid_j]
                        new_temporal_mask[i, j] = old_temporal_mask[old_i, old_j]
            self.temporal_masks = torch.nn.Parameter(torch.tensor(new_temporal_mask.flatten()), requires_grad=False)
            init_temporal_logit = torch.log(torch.tensor(0.5 / 0.5))
            self.temporal_logits = torch.nn.Parameter(
                torch.ones(N*N, requires_grad=True) * init_temporal_logit,
                requires_grad=True
            )
        print(f"Graph reconstructed with {len(self.nodes)} nodes")
        
        # Return the indices of the remaining nodes (in the new order)
        remaining_node_indices = list(range(N))
        return True, removed_node_indices, remaining_node_indices

    def check_and_remove_isolated_nodes(self, min_agents: int = 2) -> List[str]:
        """
        Checks for and removes isolated nodes from the graph.
        
        Args:
            min_agents: The minimum number of agents to keep in the graph.
            
        Returns:
            A list of IDs of the removed nodes.
        """
        isolated_node_ids = self.detect_isolated_nodes(min_agents)
        if not isolated_node_ids:
            return []
        
        # We only need the list of removed IDs, so we call remove_isolated_nodes and discard the other return values.
        self.remove_isolated_nodes(isolated_node_ids)
        return isolated_node_ids
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        # Record the mapping from node ID to index
        self.node_id_to_index[node_id] = len(self.nodes) - 1
        self.node_order.append(node_id)  # New
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        Support both homogeneous and heterogeneous LLM configurations.
        """
        for i, (agent_name, kwargs) in enumerate(zip(self.agent_names, self.node_kwargs)):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                # Assign the corresponding LLM to each agent
                if self.heterogeneous_llms and i < len(self.agent_llm_assignments):
                    kwargs["llm_name"] = self.agent_llm_assignments[i]
                else:
                    kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600, debug_log_file=None, batch_id=None) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        # ==== DEBUG: pass debug_log_file and batch_id to node.async_execute ====
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input, debug_log_file=debug_log_file, batch_id=batch_id),timeout=max_time) # output is saved in the node.outputs
                        # ==== END DEBUG ====
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min() 
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
    
    def update_masks_with_zero_extension(self, pruning_rate: float, similarity_matrix=None, mask_type='spatial') -> torch.Tensor:
        """
        Update the one-dimensional mask for graph pruning using the 0-extension algorithm
        Parameters:
        pruning_rate: The ratio of edges to prune
        similarity_matrix: Semantic similarity matrix (optional)
        mask_type: 'spatial' or 'temporal' specifying the type of mask to prune
        Returns:
        Updated masks
        """
        pruner = ZeroExtensionPruner()
        
        # Process spatial mask (one-dimensional vector)
        if self.optimized_spatial and mask_type == 'spatial':
            mask_length = len(self.spatial_masks)
            num_nodes = int(math.sqrt(mask_length))
            vertices = list(range(num_nodes))
            valid_edge_indices = torch.nonzero(self.spatial_masks, as_tuple=True)[0]
            valid_edges = []
            for idx in valid_edge_indices:
                i = idx.item() // num_nodes
                j = idx.item() % num_nodes
                if i < num_nodes and j < num_nodes:
                    valid_edges.append((i, j))
            if len(valid_edges) < 3:
                num_edges = (self.spatial_masks > 0).sum()
                num_masks = (self.spatial_masks == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate) > 0 else 1
                _edge_logits = self.spatial_logits.clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.spatial_masks[prune_idx] = 0
                return self.spatial_masks, self.temporal_masks
            edge_weights = {}
            for idx, (i, j) in enumerate(valid_edges):
                edge_index = i * num_nodes + j
                if similarity_matrix is not None:
                    if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                        edge_weights[(i, j)] = float(similarity_matrix[i, j])
                elif edge_index < len(self.spatial_logits):
                    edge_weights[(i, j)] = float(self.spatial_logits[edge_index])
            num_terminals = max(2, min(int(math.sqrt(num_nodes)), len(vertices) // 3))
            num_edges = len(valid_edges)
            prune_num_edges = int(torch.round(torch.tensor(num_edges * pruning_rate)).item()) if torch.round(torch.tensor(num_edges * pruning_rate)) > 0 else 1
            terminals = pruner.select_terminals(vertices, valid_edges, edge_weights, num_terminals)
            assignment = pruner.zero_extension(vertices, valid_edges, terminals, edge_weights, similarity_matrix=similarity_matrix)
            new_masks = torch.zeros_like(self.spatial_masks)
            for u, v in valid_edges:
                if assignment[u] == assignment[v] or u in terminals or v in terminals:
                    edge_index = u * num_nodes + v
                    if edge_index < len(new_masks):
                        new_masks[edge_index] = 1
            kept_edges = (new_masks > 0).sum().item()
            if kept_edges > num_edges - prune_num_edges:
                extra_prune = kept_edges - (num_edges - prune_num_edges)
                kept_edge_indices = torch.nonzero(new_masks, as_tuple=True)[0]
                if len(kept_edge_indices) == 0:
                    self.spatial_masks = new_masks
                    return self.spatial_masks, self.temporal_masks

                # Sort the extra edges to be pruned according to edge_score (similarity_matrix) or logits
                if similarity_matrix is not None:
                    # Sort based on edge_score
                    kept_edge_scores = torch.tensor([similarity_matrix[i.item() // num_nodes, i.item() % num_nodes] for i in kept_edge_indices])
                    sorted_indices = torch.argsort(kept_edge_scores)
                else:
                    # Sort based on logits
                    kept_edge_logits = torch.tensor([self.spatial_logits[i] for i in kept_edge_indices])
                    sorted_indices = torch.argsort(kept_edge_logits)

                # Prune the edges with the lowest scores
                prune_indices = kept_edge_indices[sorted_indices[:extra_prune]]
                new_masks[prune_indices] = 0

            self.spatial_masks = new_masks
        
        # Process temporal mask (one-dimensional vector)
        if self.optimized_temporal and mask_type == 'temporal':
            mask_length = len(self.temporal_masks)
            num_nodes = int(math.sqrt(mask_length))
            vertices = list(range(num_nodes))
            valid_edge_indices = torch.nonzero(self.temporal_masks, as_tuple=True)[0]
            valid_edges = []
            for idx in valid_edge_indices:
                i = idx.item() // num_nodes
                j = idx.item() % num_nodes
                if i < num_nodes and j < num_nodes:
                    valid_edges.append((i, j))
            if len(valid_edges) < 3:
                num_edges = (self.temporal_masks > 0).sum()
                num_masks = (self.temporal_masks == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate) > 0 else 1
                _edge_logits = self.temporal_logits.clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.temporal_masks[prune_idx] = 0
                return self.spatial_masks, self.temporal_masks
            edge_weights = {}
            for idx, (i, j) in enumerate(valid_edges):
                edge_index = i * num_nodes + j
                if similarity_matrix is not None:
                    if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                        edge_weights[(i, j)] = float(similarity_matrix[i, j])
                elif edge_index < len(self.temporal_logits):
                    edge_weights[(i, j)] = float(self.temporal_logits[edge_index])
            num_terminals = max(2, min(int(math.sqrt(num_nodes)), len(vertices) // 3))
            num_edges = len(valid_edges)
            prune_num_edges = int(torch.round(torch.tensor(num_edges * pruning_rate)).item()) if torch.round(torch.tensor(num_edges * pruning_rate)) > 0 else 1
            terminals = pruner.select_terminals(vertices, valid_edges, edge_weights, num_terminals)
            assignment = pruner.zero_extension(vertices, valid_edges, terminals, edge_weights, similarity_matrix=similarity_matrix)
            new_masks = torch.zeros_like(self.temporal_masks)
            for u, v in valid_edges:
                if assignment[u] == assignment[v] or u in terminals or v in terminals:
                    edge_index = u * num_nodes + v
                    if edge_index < len(new_masks):
                        new_masks[edge_index] = 1
            kept_edges = (new_masks > 0).sum().item()
            if kept_edges > num_edges - prune_num_edges:
                extra_prune = kept_edges - (num_edges - prune_num_edges)
                kept_edge_indices = torch.nonzero(new_masks, as_tuple=True)[0]
                if len(kept_edge_indices) == 0:
                    self.temporal_masks = new_masks
                    return self.spatial_masks, self.temporal_masks

                # Sort the extra edges to be pruned according to edge_score (similarity_matrix) or logits
                if similarity_matrix is not None:
                    # Sort based on edge_score
                    kept_edge_scores = torch.tensor([similarity_matrix[i.item() // num_nodes, i.item() % num_nodes] for i in kept_edge_indices])
                    sorted_indices = torch.argsort(kept_edge_scores)
                else:
                    # Sort based on logits
                    kept_edge_logits = torch.tensor([self.temporal_logits[i] for i in kept_edge_indices])
                    sorted_indices = torch.argsort(kept_edge_logits)

                # Prune the edges with the lowest scores
                prune_indices = kept_edge_indices[sorted_indices[:extra_prune]]
                new_masks[prune_indices] = 0

            self.temporal_masks = new_masks
        return self.spatial_masks, self.temporal_masks








