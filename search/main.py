from maze import Maze  

maze = Maze("C:/Users/DANIEL/Documents/GitHub/CS50's Introduction to Artificial Intelligence/CS50-s-Introduction-to-Artificial-Intelligence/search/maze3.txt")
print("Labirinto original:")
maze.print()

# Resolve o labirinto
maze.solve()

# Mostra o labirinto com a solução
print("Labirinto com solução:")
maze.print()

# Mostra estatísticas
print(f"Estados explorados: {maze.num_explored}")
print(f"Passos até o objetivo: {len(maze.solution[0])}")
