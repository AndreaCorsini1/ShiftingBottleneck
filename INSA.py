from graph import Graph
from time import time
from config import args, EPS


def insa(instance: dict):
    """
    INSertion Algorithm for Job Shop.

    Args:
        instance: Job Shop instance.
    Returns:
        Solution as a matrix and respective Directed Acyclic Graph.
    """
    num_j, num_m = instance['j'], instance['m']
    costs, machines = instance['costs'], instance['machines']
    dag = Graph(costs, machines, num_j, num_m)

    # Initialize solution with operations of the longest job
    job = costs.sum(-1).argmax().item()
    sol = [[] for _ in range(num_m)]
    n = job * num_m
    for i in range(num_m):
        sol[machines[job, i]].append(n + i)

    # Remaining operations to schedule
    operations = sorted([
        (costs[j, i], j * num_m + i, machines[j, i])
        for j in range(num_j) for i in range(num_m) if j != job
    ], key=lambda el: el[0], reverse=True)

    # Schedule operations in decreasing order of their processing time
    for _, n, m in operations:
        machine = sol[m]
        pos, best_d = None, float('inf')
        for i, succ in enumerate(machine):
            #
            if i == 0:
                dag.add_arc(n, succ)
                d = dag.longest_through(n)
                if d is not None and d < best_d - EPS:
                    best_d, pos = d, i
                dag.remove_arc(n, succ)
            else:
                pred = machine[i - 1]
                dag.remove_arc(pred, succ)
                dag.add_arc(pred, n)
                dag.add_arc(n, succ)
                d = dag.longest_through(n)
                if d is not None and d < best_d - EPS:
                    best_d, pos = d, i
                dag.remove_arc(pred, n)
                dag.remove_arc(n, succ)
                dag.add_arc(pred, succ)

        # Last pos
        dag.add_arc(machine[-1], n)
        d = dag.longest_through(n)
        if d is None or d > best_d - EPS:
            dag.remove_arc(machine[-1], n)
            if pos == 0:
                dag.add_arc(n, machine[0])
            else:
                dag.remove_arc(machine[pos - 1], machine[pos])
                dag.add_arc(machine[pos - 1], n)
                dag.add_arc(n, machine[pos])
            machine.insert(pos, n)
        else:
            machine.append(n)
    #
    return sol, dag


if __name__ == '__main__':
    print(f'CONFIG: {args}')
    import pandas as pd
    from inout import load_dataset
    instances = load_dataset(f'benchmarks/{args.name}', basic=True)

    # 4 DEBUGGING
    # from inout import read_jsp
    # instances = [read_jsp(f'benchmarks/LA/la29.jsp')]
    # instances = [read_jsp(f'benchmarks/DMU/dmu75.jsp')]
    # instances = [read_jsp(f'benchmarks/TA/ta78.jsp')]

    #
    for ins in instances:
        print(f'Solving {ins["name"]} ({ins["shape"]}): UB = {ins["makespan"]}')
        st = time()
        sol, g = insa(ins)
        tt = time() - st

        # Compute final makespan
        ms, _, _ = g.makespan()
        if ms is None:
            print(f"ERROR: {ins['name']}: unfeasible solution found")
        print(f"--> FINAL MS={ms}")

        # Save results
        df = pd.DataFrame([{'name': ins['name'],
                            'ub': ins['makespan'],
                            'time': tt,
                            'ms': ms,
                            'gap': (ms / ins['makespan'] - 1) * 100}])
        df.to_csv(f'output/{args.name}_INSA.csv',
                  mode='a',
                  sep=',',
                  index=False,
                  header=False)
