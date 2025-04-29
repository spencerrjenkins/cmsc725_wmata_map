import random
import numpy as np

def is_urban_core(node_pos, core_bounds):
    x, y = node_pos
    xmin, ymin, xmax, ymax = core_bounds
    return xmin <= x <= xmax and ymin <= y <= ymax

def classify_nodes(positions, core_bounds):
    return {n: 'urban' if is_urban_core(pos, core_bounds) else 'suburb' for n, pos in positions.items()}

def route_pattern_score(route, node_types):
    # node_types: dict of node_id -> 'urban' or 'suburb'
    if len(route) < 3:
        return 0
    start_type = node_types[route[0]]
    end_type = node_types[route[-1]]
    passes_urban = any(node_types[n] == 'urban' for n in route[1:-1])
    if start_type == 'suburb' and end_type == 'suburb' and passes_urban:
        return 1  # or a higher bonus
    return 0

def initialize_population(graph, positions, population_size, num_routes, min_distance, max_distance):
    """Initialize a population of candidate sets of routes."""
    from funcs import perform_walks
    population = []
    for _ in range(population_size):
        traversed_edges = set()
        complete_traversed_edges = []
        walks, _, _ = perform_walks(
            graph, positions, num_walks=num_routes, min_distance=min_distance, max_distance=max_distance,
            traversed_edges=traversed_edges, complete_traversed_edges=complete_traversed_edges
        )
        if walks and len(walks) == num_routes:
            population.append(walks)
    return population

def fitness(route_set, positions, kde, radius=2000, node_types=None):
    from funcs import score_walk_by_kde
    route_scores = [score_walk_by_kde(walk, positions, kde, radius) for walk in route_set]
    pattern_bonus = sum(route_pattern_score(walk, node_types) for walk in route_set)
    unique_nodes = set()
    for walk in route_set:
        unique_nodes.update(walk)
    coverage_score = len(unique_nodes)
    return sum(route_scores) + 0.1 * coverage_score + 10 * pattern_bonus  # Tune weights as needed

def selection(population, fitnesses, num_selected):
    """Select the best individuals based on fitness."""
    selected_indices = np.argsort(fitnesses)[-num_selected:]
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    """Crossover: swap random routes between two sets."""
    num_routes = len(parent1)
    idx = random.randint(0, num_routes - 1)
    child1 = parent1[:idx] + parent2[idx:]
    child2 = parent2[:idx] + parent1[idx:]
    return child1, child2

def mutate(route_set, graph, mutation_rate=0.1):
    """Mutate a set of routes by mutating one route."""
    new_set = [route[:] for route in route_set]
    if random.random() < mutation_rate:
        idx = random.randint(0, len(new_set) - 1)
        walk = new_set[idx]
        if len(walk) > 2:
            node_idx = random.randint(1, len(walk) - 2)
            node = walk[node_idx]
            neighbors = list(graph.neighbors(node))
            if neighbors:
                walk[node_idx] = random.choice(neighbors)
        new_set[idx] = walk
    return new_set

def genetic_algorithm(
    graph, positions, kde, num_routes=3, population_size=20, generations=30,
    min_distance=20000, max_distance=40000, radius=2000, mutation_rate=0.1,
    core_bounds=None, caller=lambda a: None
):
    """
    GA for best set of routes, prioritizing suburb-urban-suburb patterns.
    Pass core_bounds=(xmin, ymin, xmax, ymax) for urban core.
    """
    # Classify nodes as 'urban' or 'suburb'
    if core_bounds is not None:
        node_types = classify_nodes(positions, core_bounds)
    else:
        # Default: treat all as 'suburb'
        node_types = {n: 'suburb' for n in positions}

    population = initialize_population(graph, positions, population_size, num_routes, min_distance, max_distance)
    for gen in range(generations):
        fitnesses = [fitness(route_set, positions, kde, radius, node_types) for route_set in population]
        # Selection
        selected = selection(population, fitnesses, max(2, population_size // 2))
        # Crossover
        children = []
        while len(children) < population_size:
            parents = random.sample(selected, 2)
            child1, child2 = crossover(parents[0], parents[1])
            children.extend([child1, child2])
        # Mutation
        population = [mutate(child, graph, mutation_rate) for child in children[:population_size]]
        caller(gen)
    # Final selection
    fitnesses = [fitness(route_set, positions, kde, radius, node_types) for route_set in population]
    best_idx = int(np.argmax(fitnesses))
    return population[best_idx], fitnesses[best_idx]