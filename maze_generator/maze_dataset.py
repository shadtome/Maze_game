from torch.utils.data import Dataset
from maze_dataset.generation import LatticeMazeGenerators

class Maze_dataset(Dataset):
    def __init__(self,num_mazes,shape = (10,10)):
        self.mazes = [self.generate_maze(shape) for _ in range(num_mazes)]


    def generate_maze(self,shape):
        maze = LatticeMazeGenerators.gen_dfs(
            grid_shape = shape,
            lattice_dim=2,
            accessible_cells=None,
            max_tree_depth=None,
            start_coord=None,
        )
        return maze

    def __len__(self):
        return len(self.mazes)
    
    def __getitem__(self,idx):
        return self.mazes[idx]