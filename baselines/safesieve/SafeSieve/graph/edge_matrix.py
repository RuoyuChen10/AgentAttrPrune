import numpy as np

class EdgeMatrixManager:
    def __init__(self, N, similarity_matrix, alpha=1.0, beta=1.0, epsilon=1e-6):
        self.N = N
        self.similarity_matrix = similarity_matrix
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.complementarity_count = np.zeros((N, N), dtype=np.float32)
        self.complementarity_total = 0.0

    def update_matrix_size(self, new_N, old_node_indices=None, new_node_indices=None):
        """
        Update the matrix size when nodes are removed
        
        Args:
            new_N: New number of nodes
            old_node_indices: Mapping from old node indices to new node indices (optional)
            new_node_indices: List of new node indices (optional)
        """
        if new_N == self.N:
            return  # No change in size
        
        print(f"EdgeMatrixManager: Updating matrix size from {self.N}x{self.N} to {new_N}x{new_N}")
        
        # Create new matrices
        new_complementarity_count = np.zeros((new_N, new_N), dtype=np.float32)
        new_similarity_matrix = np.zeros((new_N, new_N), dtype=np.float32)
        
        # If there is a node index mapping, retain the corresponding data
        if old_node_indices is not None and new_node_indices is not None:
            for old_i, new_i in enumerate(new_node_indices):
                if new_i < new_N:
                    for old_j, new_j in enumerate(new_node_indices):
                        if new_j < new_N:
                            if old_i < self.N and old_j < self.N:
                                new_complementarity_count[new_i, new_j] = self.complementarity_count[old_i, old_j]
                                new_similarity_matrix[new_i, new_j] = self.similarity_matrix[old_i, old_j]
        
        # Update values
        self.N = new_N
        self.complementarity_count = new_complementarity_count
        self.similarity_matrix = new_similarity_matrix
        # Recalculate total
        self.complementarity_total = self.complementarity_count.sum()

    def remove_nodes_and_update_matrix(self, removed_node_indices, remaining_node_indices):
        """
        Remove specified nodes and update the matrix, preserving historical accumulated values
        
        Args:
            removed_node_indices: List of indices of nodes to be removed
            remaining_node_indices: List of indices of nodes to be retained
        """
        if not removed_node_indices:
            return
        
        new_N = len(remaining_node_indices)
        print(f"EdgeMatrixManager: Removing nodes {removed_node_indices}, updating matrix from {self.N}x{self.N} to {new_N}x{new_N}")
        
        # Create new matrices
        new_complementarity_count = np.zeros((new_N, new_N), dtype=np.float32)
        new_similarity_matrix = np.zeros((new_N, new_N), dtype=np.float32)
        
        # Map retained node data to the new matrix
        for i, old_idx in enumerate(remaining_node_indices):
            for j, old_jdx in enumerate(remaining_node_indices):
                if old_idx < self.N and old_jdx < self.N:
                    new_complementarity_count[i, j] = self.complementarity_count[old_idx, old_jdx]
                    new_similarity_matrix[i, j] = self.similarity_matrix[old_idx, old_jdx]
        
        # Update values
        self.N = new_N
        self.complementarity_count = new_complementarity_count
        self.similarity_matrix = new_similarity_matrix
        # Recalculate total
        self.complementarity_total = self.complementarity_count.sum()

    def update_complementarity(self, adj_matrix, solved):
        if solved:
            # Ensure adj_matrix size matches current N
            if adj_matrix.shape[0] != self.N or adj_matrix.shape[1] != self.N:
                print(f"Warning: adj_matrix size {adj_matrix.shape} doesn't match current N={self.N}")
                # If the size does not match, adjust adj_matrix size
                if adj_matrix.shape[0] > self.N:
                    adj_matrix = adj_matrix[:self.N, :self.N]
                elif adj_matrix.shape[0] < self.N:
                    # Expand adj_matrix to the correct size
                    new_adj = np.zeros((self.N, self.N))
                    new_adj[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
                    adj_matrix = new_adj
            self.complementarity_count += adj_matrix
            self.complementarity_total += adj_matrix.sum()

    def update_beta_weight(self, new_beta):
        """Update complementarity weight"""
        self.beta = new_beta

    def get_complementarity(self):
        N = self.N
        # Fix denominator calculation: when total is 0, epsilon should be used as the denominator
        if self.complementarity_total == 0:
            return np.zeros((N, N))
        return (self.complementarity_count + self.epsilon) / (self.complementarity_total + N*N*self.epsilon)

    def get_edge_score(self):
        sim = self.similarity_matrix
        comp = self.get_complementarity()
        edge_score = self.alpha * sim + self.beta * comp
        # Normalize to [0,1]
        edge_score = (edge_score - edge_score.min()) / (edge_score.max() - edge_score.min() + 1e-8)
        return edge_score

    def get_edge_score_with_debug(self):
        """Return edge score and debug information"""
        sim = self.similarity_matrix
        comp = self.get_complementarity()
        edge_score = self.alpha * sim + self.beta * comp
        # Normalize to [0,1]
        edge_score = (edge_score - edge_score.min()) / (edge_score.max() - edge_score.min() + 1e-8)
        
        return edge_score, {
            'similarity_matrix': sim,
            'complementarity_matrix': comp,
            'alpha': self.alpha,
            'beta': self.beta,
            'raw_edge_score': self.alpha * sim + self.beta * comp
        }

    def get_debug_info(self):
        """Return debug information"""
        return {
            'complementarity_count': self.complementarity_count,
            'complementarity_total': self.complementarity_total,
            'complementarity_matrix': self.get_complementarity()
        }

    def reset(self):
        self.complementarity_count.fill(0)
        self.complementarity_total = 0.0 