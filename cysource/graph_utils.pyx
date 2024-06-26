# cython: profile=False
# cython: language_level=3
"""
Cython implementation of computationally expensive functions.
"""
import numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef compute_makespan(const long[:, ::1] g_successors,
                       const long[:, ::1] g_predecessors,
                       const float[::1] costs,
                       long[::1] cp):
    """
    Compute the makespan (a.k.a. the longest weighted path in the graph)
    
    Args:
        g_successors: (2D array)
            The successors of each node. Each row represents a different node.
            The first column gives the successor on the same job, while the 
            second column gives the successor on the same machine.
        g_predecessors: (2D array)
            The number of ingoing arcs to each node.
        costs: (1D array)
            The cost of each node.
        cp: (1D array)
            The output critical path, if the solution is feasible.
    Returns:
        1. The makespan, if the graph is cyclic None.
    """
    cdef float distances[100000]    # Longest path to each node
    cdef int stack[100000]          # LIFO for visiting graph
    cdef int in_degree[100000]      # Current in degree of nodes
    cdef long preds[100000]          # The predecessors for computing paths
    cdef int idx = -1
    cdef int num_nodes = len(costs)
    cdef int i, n
    cdef int last = 0
    cdef float makespan = 0
    cdef int seen = 0
    # Initialize data structures
    for i in range(num_nodes):
        distances[i] = costs[i]
        stack[i] = -1
        preds[i] = -1
        in_degree[i] = 1 if g_predecessors[i, 0] >= 0 else 0
        if g_predecessors[i, 1] >= 0:
            in_degree[i] += 1
        elif in_degree[i] == 0:
            idx += 1
            preds[i] = i
            stack[idx] = i
    # Explore the whole graph
    while idx >= 0:
        seen += 1
        i = stack[idx]
        stack[idx] = -1
        idx -= 1

        # Successor on the same job, if any
        n = g_successors[i, 0]
        if n >= 0:
            # Update longest distance
            if distances[n] < distances[i] + costs[n] - 0.0005:
                distances[n] = distances[i] + costs[n]
                preds[n] = i
                # Update the makespan
                if distances[n] - 0.0005 > makespan:
                    makespan = distances[n]
                    last = n
            in_degree[n] -= 1
            # Put successor in the stack if all predecessors have been seen
            if in_degree[n] == 0:
                idx += 1
                stack[idx] = n

        # Successor on the same machine, if any
        n = g_successors[i, 1]
        if n >= 0:
            # Update longest distance
            if distances[n] < distances[i] + costs[n] - 0.0005:
                distances[n] = distances[i] + costs[n]
                preds[n] = i
                # Update the makespan
                if distances[n] - 0.0005 > makespan:
                    makespan = distances[n]
                    last = n
            in_degree[n] -= 1
            # Put successor in the stack if all predecessors have been seen
            if in_degree[n] == 0:
                idx += 1
                stack[idx] = n
    #
    if seen == num_nodes:
        cp[0] = last
        idx = 1
        while last != preds[last]:
            last = preds[last]
            cp[idx] = last
            idx += 1
        cp[idx] = -1
        return makespan, distances
    else:
        # print(f"Num seen nodes: {seen}/{num_nodes}.")
        return None, distances


@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
cpdef longest_to(int target,
                 const long[:, ::1] g_successors,
                 const long[:, ::1] g_predecessors,
                 const float[::1] costs):
    """
    Compute the length of the longest path through a target node.

    Args:
        target: (int)
            The target node.
        g_successors: (2D array)
            The successors of each node. Each row represents a different node.
            The first column gives the successor on the same job, while the 
            second column gives the successor on the same machine.
        g_predecessors: (2D array)
            The number of ingoing arcs to each node.
        costs: (1D array)
            The cost of each node.
    Returns:
        1. The legnth of the path, if the graph is cyclic None.
    """
    if g_predecessors[target, 0] < 0 and g_predecessors[target, 1] < 0:
        return 0

    cdef float distances[100000]    # Longest path to each node
    cdef int stack[100000]          # LIFO for visiting graph
    cdef int in_degree[100000]      # Current in degree of nodes
    cdef int idx = -1
    cdef int num_nodes = len(costs)
    cdef int i, n
    #
    for i in range(num_nodes):
        distances[i] = costs[i]
        stack[i] = -1
        in_degree[i] = 1 if g_predecessors[i, 0] >= 0 else 0
        if g_predecessors[i, 1] >= 0:
            in_degree[i] += 1
        elif in_degree[i] == 0:
            idx += 1
            stack[idx] = i
    #
    while idx >= 0:
        i = stack[idx]
        stack[idx] = -1
        idx -= 1

        # Successor on the same job, if any
        n = g_successors[i, 0]
        if n >= 0:
            # Update longest distance
            if distances[n] < distances[i] + costs[n] - 0.0005:
                distances[n] = distances[i] + costs[n]
            in_degree[n] -= 1
            # Put successor in the stack if all predecessors have been seen
            if in_degree[n] == 0:
                if n == target:
                    return distances[target] - costs[target]
                idx += 1
                stack[idx] = n

        # Successor on the same machine, if any
        n = g_successors[i, 1]
        if n >= 0:
            # Update longest distance
            if distances[n] < distances[i] + costs[n] - 0.0005:
                distances[n] = distances[i] + costs[n]
            in_degree[n] -= 1
            # Put successor in the stack if all predecessors have been seen
            if in_degree[n] == 0:
                if n == target:
                    return distances[target] - costs[target]
                idx += 1
                stack[idx] = n
    # Cannot reach this point
    print("ERROR: not visited node", target, in_degree[i], distances[i])


@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
cpdef longest_from(int target,
                   const long[:, ::1] g_successors,
                   const long[:, ::1] g_predecessors,
                   const float[::1] costs):
    """
    Compute the length of the longest path through a target node.

    Args:
        target: (int)
            The target node.
        g_successors: (2D array)
            The successors of each node. Each row represents a different node.
            The first column gives the successor on the same job, while the 
            second column gives the successor on the same machine.
        g_predecessors: (2D array)
            The number of ingoing arcs to each node.
        costs: (1D array)
            The cost of each node.
    Returns:
        1. The legnth of the path, if the graph is cyclic None.
    """
    if g_successors[target, 0] < 0 and g_successors[target, 1] < 0:
        return 0

    cdef float distances[100000]    # Longest path to each node
    cdef int stack[100000]          # LIFO for visiting graph
    cdef int num_nodes = len(costs)
    cdef int i, n
    cdef int idx = -1
    cdef float path_length = 0
    #
    for i in range(num_nodes):
        # distances[i] = costs[i]
        distances[i] = 0
        stack[i] = -1
        if i == target:
            idx += 1
            stack[idx] = i
    #
    while idx >= 0:
        i = stack[idx]
        stack[idx] = -1
        idx -= 1

        # Successor on the same job, if any
        n = g_successors[i, 0]
        if n >= 0:
            # Update longest distance
            if distances[n] < distances[i] + costs[n] - 0.0005:
                distances[n] = distances[i] + costs[n]
                if distances[n] - 0.0005 > path_length:
                    path_length = distances[n]
            # Put successor in the stack
            idx += 1
            stack[idx] = n

        # Successor on the same machine, if any
        n = g_successors[i, 1]
        if n >= 0:
            # Update longest distance
            if distances[n] < distances[i] + costs[n] - 0.0005:
                distances[n] = distances[i] + costs[n]
                if distances[n] - 0.0005 > path_length:
                    path_length = distances[n]
            # Put successor in the stack
            idx += 1
            stack[idx] = n

    return path_length


@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
cpdef longest_path(int target,
                   const long[:, ::1] g_successors,
                   const long[:, ::1] g_predecessors,
                   const float[::1] costs):
    """
    Compute the length of the longest path through a target node.

    Args:
        target: (int)
            The target node.
        g_successors: (2D array)
            The successors of each node. Each row represents a different node.
            The first column gives the successor on the same job, while the 
            second column gives the successor on the same machine.
        g_predecessors: (2D array)
            The number of ingoing arcs to each node.
        costs: (1D array)
            The cost of each node.
    Returns:
        1. The legnth of the path, if the graph is cyclic None.
    """
    cdef float distances[100000]    # Longest path to each node
    cdef int stack[100000]          # LIFO for visiting graph
    cdef int in_degree[100000]      # Current in degree of nodes
    cdef int idx = -1
    cdef int num_nodes = len(costs)
    cdef int i, j, n
    cdef int seen = 0
    cdef float path_length = 0
    #
    for i in range(num_nodes):
        distances[i] = costs[i]
        stack[i] = -1
        in_degree[i] = 1 if g_predecessors[i, 0] >= 0 else 0
        if g_predecessors[i, 1] >= 0:
            in_degree[i] += 1
        elif in_degree[i] == 0:
            idx += 1
            stack[idx] = i
    #
    while idx >= 0:
        seen += 1
        i = stack[idx]
        stack[idx] = -1
        idx -= 1
        # Reset the distances of all but target node
        if i == target:
            for j in range(num_nodes):
                if j != target:
                    distances[j] = 0
                else:
                    distances[j] += 100000
                    path_length = distances[j]
        # For each successor
        for j in range(2):
            n = g_successors[i, j]
            if n >= 0:
                # Update longest distance
                if distances[n] < distances[i] + costs[n] - 0.0005:
                    distances[n] = distances[i] + costs[n]
                    # Update makespan
                    if distances[n] - 0.0005 > path_length:
                        path_length = distances[n]
                in_degree[n] -= 1
                # Put successor in the stack if all predecessors have been seen
                if in_degree[n] == 0:
                    idx += 1
                    stack[idx] = n
    #
    if seen == num_nodes:
        return path_length - 100000
    else:
        return None
