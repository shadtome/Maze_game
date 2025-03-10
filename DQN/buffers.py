import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree for sum priorities
        self.data = np.array([None]*capacity)  # Store experiences
        self.size = 0
        self.write = 0
    
    def _propagate(self, idx, change):
        """Update parent nodes to reflect new priority sums."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def add(self, priority, data):
        """Add new experience with given priority."""
        idx = self.write + self.capacity - 1  # Get tree index
        self.data[self.write] = data  # Store experience
        self.update(idx, priority)  # Update priority in tree

        self.write = (self.write + 1) % self.capacity  # Cycle index
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority in tree and propagate change."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _retrieve(self, idx, s):
        """Find sample index corresponding to priority value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # Leaf node
            return idx

        if s <= self.tree[left]:  # Go left
            return self._retrieve(left, s)
        else:  # Go right
            return self._retrieve(right, s - self.tree[left])

    def sample(self, s):
        """Sample experience based on priority."""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def total_priority(self):
        return self.tree[0]  # Root node stores total priority
    
    def __len__(self):
        return self.size
    

class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Prioritization factor
        self.start_beta = beta
        self.beta = beta  # Importance sampling correction factor
        self.beta_increment = beta_increment  # How fast beta increases
        self.epsilon = 1e-5  # Small value to avoid zero priority
    
    def add(self, experience, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        batch = []
        segment = self.tree.total_priority() / batch_size
        min_prob = float('inf')  # For normalizing IS weights
        
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, experience = self.tree.sample(s)
            if experience is None:
                continue
            batch.append((idx, experience, priority))
            min_prob = min(min_prob, priority / self.tree.total_priority())

        # Compute IS weights
        probs = np.array([p for _, _, p in batch]) / self.tree.total_priority()
        is_weights = (probs * len(self.tree)) ** (-self.beta)
        is_weights /= max(is_weights)  # Normalize so max weight = 1

        # Append IS weights to each sample in the batch
        #batch = [(idx, exp, p, w) for (idx, exp, p), w in zip(batch, is_weights)]
        
        return batch, is_weights
    
    def step(self,frame,n_frames):
        linear_decay = self.start_beta*((n_frames - frame)/n_frames) + 1.0*(frame/n_frames)
        self.beta = min(1.0,linear_decay)
    
    def update_priority(self, idx, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)