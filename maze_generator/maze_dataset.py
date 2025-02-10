from torch.utils.data import Dataset
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.generation import generators

class Maze_dataset(Dataset):
    def __init__(self,num_mazes,shape = (10,10),maze_type = 'dfs'):
        """Used to generate mazes
            num_mazes: the number of mazes to generate
            shape: shape of the maze (h,w)
            type: the type of maze generation.
                    - 'dfs': depth first search
                    - 'wilson': wilson's algorithm
                    - 'percolation': percolation maze generation
                    - 'prim': """
        self.maze_type = maze_type
        self.mazes = [self.generate_maze(shape) for _ in range(num_mazes)]
        LatticeMazeGenerators.gen_percolation
        LatticeMazeGenerators.gen_wilson
        LatticeMazeGenerators.gen_prim

    def generate_maze(self,shape):
        """Generate mazes based on the type of maze generation
        shape: shape of the maze (h,w)"""
        if self.maze_type == 'dfs':
            maze = LatticeMazeGenerators.gen_dfs(
                grid_shape = shape,
                lattice_dim=2,
                accessible_cells=None,
                max_tree_depth=None,
                start_coord=None,
            )
            return maze
        if self.maze_type == 'wilson':
            maze = LatticeMazeGenerators.gen_wilson(
                grid_shape=shape
            )
            return maze
        if self.maze_type == 'percolation':
            maze = LatticeMazeGenerators.gen_percolation(
                grid_shape=shape,
                p=0.4,
                lattice_dim = 2,
                start_coord=None,
            )
            return maze
        if self.maze_type == 'prim':
            maze = LatticeMazeGenerators.gen_prim(
                grid_shape=shape,
                lattice_dim=2,
                accessible_cells=None,
                max_tree_depth=None,
                do_forks=True,
                start_coord=None
            )

    def __len__(self):
        return len(self.mazes)
    
    def __getitem__(self,idx):
        return self.mazes[idx]