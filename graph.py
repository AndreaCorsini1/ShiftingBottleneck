"""
Utils for making and drawing Networkx graph.
"""
import networkx as nx
import numpy as np
from cysource import graph_utils as cutils


class Graph(object):
    """
    Directed acyclic graph for the Job Shop Scheduling with Cython extension
    for computing the makespan.

    Args:
        costs: Rows correspond to jobs and columns to the cost of
            operations.
        machines: Rows correspond to jobs and columns to the machine of
            operations.
        num_j: Number of jobs.
        num_m: Number of machines.
    """
    def __init__(self, costs: np.ndarray, machines: np.ndarray,
                 num_j: int = None, num_m: int = None):
        if num_j is None:
            self.num_j, self.num_m = costs.shape
        else:
            self.num_j = num_j
            self.num_m = num_m
        self.num_nodes = self.num_j * self.num_m

        self.costs = np.ascontiguousarray(costs.reshape(-1), dtype=np.float32)
        self.machines = np.ascontiguousarray(machines.reshape(-1), dtype=np.int32)

        self.predecessors = -np.ones((self.num_nodes, 2), dtype=np.int64)
        self.successors = -np.ones((self.num_nodes, 2), dtype=np.int64)

        # Conjunctive arcs
        # Note that conjunctive arcs are always on the first column of
        # self.successors and self.predecessors
        for job in range(self.num_j):
            for idx in range(self.num_m - 1):
                n = job * self.num_m + idx
                self.successors[n, 0] = n + 1
                self.predecessors[n + 1, 0] = n

        # 4 MAKESPAN
        self._cp = -np.ones(self.num_nodes + 1, dtype=np.int64)

    def __str__(self):
        return f"DiGraph(Num nodes: {self.num_nodes})"

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """
        Make a deepcopy of the graph.

        Returns:
            The new graph
        """
        new_g = Graph(self.costs, self.machines, self.num_j, self.num_m)
        new_g.predecessors = np.array(self.predecessors, copy=True, order='C')
        new_g.successors = np.array(self.successors, copy=True, order='C')
        return new_g

    def add_arc(self, s: int, e: int):
        """
        Add a disjunctive arc.

        Args:
            s: Source node.
            e: Destination node.
        Returns:
            None
        """
        self.successors[s, 1] = e
        self.predecessors[e, 1] = s

    def remove_arc(self, s: int, e: int):
        """
        Remove a disjunctive arc.

        Args:
            s: Source node.
            e: Destination node.
        Returns:
            None
        """
        self.successors[s, 1] = -1
        self.predecessors[e, 1] = -1

    def from_solution(self, solution: np.ndarray):
        """
        Make the graph from a matrix-like solution.

        Args:
            solution: Rows correspond to machines and columns to operations in
                the machine.
        Returns:
             The graph.
        """
        for nodes in solution:
            for s, e in zip(nodes[:-1], nodes[1:]):
                self.successors[s, 1] = e
                self.predecessors[e, 1] = s

            self.predecessors[nodes[0], 1] = -1
            self.successors[nodes[-1], 1] = -1

        return self

    def to_solution(self, dtype = np.int64):
        """
        Make a matrix-like solution from the graph.

        Args:
            dtype: data type.
        Returns:
            Matrix (list of list)
        """
        sol = np.zeros((self.num_m, self.num_j), dtype=dtype)
        for m, perm in enumerate(sol):
            next_n = None
            for n in np.argwhere(self.machines == m).squeeze():
                if self.predecessors[n, 1] == -1 and self.successors[n, 1] >= 0:
                    next_n = n
                    break
            assert next_n is not None, \
                f"Error: no starting node form machine {m}"
            #
            idx = 1
            perm[0] = next_n
            while self.successors[next_n, 1] >= 0:
                next_n = self.successors[next_n, 1]
                perm[idx] = next_n
                idx += 1
        return sol

    def to_networkx(self):
        """
        Transform the graph into Networkx DiGraph.

        Returns:
            nx.Digraph
        """
        g = nx.DiGraph()

        # Make nx digraph
        for j in range(self.num_j):
            for i in range(self.num_m):
                n = j * self.num_m + i
                p = self.costs[n]
                g.add_node(n, job=j, op=i, cost=p, machine=self.machines[n])

                # Add arcs
                if self.successors[n, 0] >= 0:
                    g.add_edge(n, self.successors[n, 0], cost=p)
                if self.successors[n, 1] >= 0:
                    g.add_edge(n, self.successors[n, 1], cost=p)
        return g

    def makespan(self, reverse: bool = False):
        """
        Compute the makespan on the input graph.
        We DO NOT reverse the critical path

        Returns:
            The makespan and the critical path (ndarray).
        """
        #
        _cp = self._cp
        _cp[:] = -1
        if reverse:
            # This is useful to compute the longest path from nodes
            ms, dist = cutils.compute_makespan(self.predecessors, self.successors,
                                               self.costs, _cp)
        else:
            ms, dist = cutils.compute_makespan(self.successors, self.predecessors,
                                               self.costs, _cp)
        if ms is None:
            cycle = {s: e for s, e, _ in
                     nx.find_cycle(self.to_networkx(), orientation='original')}
            # print(dist[:self.num_nodes])
            # self.debug()
            return None, cycle, dist[:self.num_nodes]
        #
        return ms, _cp[_cp >= 0], dist[:self.num_nodes]

    def longest_through(self, target: int):
        """
        Compute the longest path through a target node.

        Args:
            target: Index of the target node.
        Returns:
            The length of the path.
        """
        return cutils.longest_path(target, self.successors,
                                   self.predecessors, self.costs)

    def longest_from(self, target: int):
        """
        Compute the longest path from a target node.

        Args:
            target: Index of the target node.
        Returns:
            The length of the path.
        """
        return int(cutils.longest_from(target, self.successors,
                                       self.predecessors, self.costs) + 0.1)

    def debug(self):
        # print('EDGE LIST:')
        # for i in range(self.num_nodes):
        #     print(f"  {i}: C={self.costs[i]}, "
        #           f"SUCC=({self.successors[i, 0]:3}, {self.successors[i, 1]:3}), ")
                  # f"PRED=({self.predecessors[i, 0]:3}, {self.predecessors[i, 1]:3})")
        try:
            cycle = nx.find_cycle(self.to_networkx(), orientation='original')
            print('Detected cycle:\n', cycle)
        except nx.NetworkXNoCycle:
            print('No cycle!')


def compute_makespan_nx(dag: nx.DiGraph, attr: str = 'cost',
                        return_distances: bool = False):
    """
    Compute the makespan on the input graph.

    Args:
        dag: graph shaping the solution.
        attr: attribute to sum for computing the makespan.
        return_distances:
    :return: if the graph is feasible (i.e. it is a dag and it hasn't loops)
        the function returns the makespan and the critical path; if the graph
        is not feasible, the function returns None, None
    """
    predecessor = {}
    distances = {}
    try:
        for n in nx.topological_sort(dag):
            if n not in distances:
                predecessor[n] = -1
                distances[n] = dag.nodes[n][attr]

            for succ in dag.successors(n):
                succ_dist = distances[n] + dag.nodes[succ][attr]
                if succ not in distances or succ_dist > distances[succ]:
                    distances[succ] = succ_dist
                    predecessor[succ] = n
    except nx.NetworkXException:
        return None, None
    # print(distances)
    max_node = max(distances, key=distances.get)
    makespan = distances[max_node]
    # print(predecessor)
    critical_path = []
    while max_node != -1:
        critical_path.append(max_node)
        max_node = predecessor[max_node]

    if return_distances:
        return makespan, list(reversed(critical_path)), distances
    else:
        return makespan, list(reversed(critical_path))


def check_solution(ins: dict, sol: np.ndarray, makespan: float = None,
                   eps: float = 0.00025):
    """
    Check the correctness of a solution.
    A solution is correct if:
        - The previous operation on the job completes before the start time of 
          the consecutive op.
        - The previous operation on the solution (machine) completes before 
          the start time of the consecutive op.
        - One of the idle times between consecutive operations on the job and 
          the solution must be 0.
    
    Args:
        ins: JSP instance.
        sol: Solution where each row gives the permutation of operations
            on a machine.
        makespan: The makespan returned by the pointer net strategy (optional).
        eps: Tolerance for comparisons.
    Returns:
        None
    """
    num_j = ins['j']
    num_m = ins['m']
    dag = Graph(ins['costs'].cpu().numpy(), ins['machines'].cpu().numpy())
    dag.from_solution(sol)
    ms, _, times = compute_makespan_nx(dag.to_networkx(), return_distances=True)
    # Check feasibility
    ms, _, _ = g.makespan()
    if ms is None:
        raise RuntimeError("ERR0: cycle detected! ",
                           nx.find_cycle(g.to_networkx()))
    # Mismatching makespan
    if makespan is not None and abs(ms - makespan) > eps:
        raise RuntimeError(f"Mismatching makespan on instance {ins['name']}")
    # Check jobs
    for j in range(num_j):
        for i in range(1, num_m):
            idx = j * num_m + i
            if times[idx - 1] - eps > times[idx] - dag.costs[idx]:
                raise ValueError(f"ERR0: OP-{idx-1} completes after the "
                                 f"start of OP-{idx}")
    # Check machines
    for machine in sol:
        for i in range(1, num_m):
            idx = machine[i]
            st = times[idx] - dag.costs[idx]
            m_pred = machine[i - 1]
            # Check start time of op is after completion of previous op
            idle = st - times[m_pred]
            if idle < -eps:
                raise ValueError(f"ERR1: OP {m_pred} completes after the "
                                 f"start of OP {idx}")
            # one between machine and job predecessor must complete an
            # instant before start time of op
            if idle > eps:
                j_pred = idx - 1 if idx % num_m else None
                if j_pred is None:
                    raise ValueError(f"IDLE: OP {idx} can start earlier")
                idle = st - times[j_pred]
                if idle > eps:
                    raise ValueError(f"IDLE: OP {idx} can start earlier")


if __name__ == '__main__':
    # 4 TESTING
    from inout import load_dataset
    for _ins in load_dataset("./benchmarks/TA", use_cached=True):
        g = Graph(_ins['costs'].numpy(), _ins['machines'].numpy())
        g.from_solution(_ins['sol'].numpy())
        #
        ms, cp, _ = g.makespan()
        if int(ms) != int(_ins['makespan']):
            print(_ins['name'], _ins['path'], _ins['makespan'], ms)
