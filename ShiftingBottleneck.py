import numpy as np
from graph import Graph
from time import time
from ortools.sat.python import cp_model
from collections import namedtuple
from tqdm import tqdm
from config import args, EPS, DEBUG

# Named tuples for cleaner code
CPTask = namedtuple('CPTask', 'start end interval tail')
Task = namedtuple('Task', 'p head tail')


def solve_lmax(tasks: dict, precs: list = None, erd_init: bool = False,
               horizon: int = 20000):
    """
    Solve Lmax problem on a single machine with release dates and tails
    after each job.

    Args:
        tasks: tasks/operations to be scheduled.
        precs: list of precedences that are not allowed in solutions (optional).
        horizon: scheduling horizon.
    Return:
         Lmax value.
         Machine permutation.
         Solution as a list of tuple (start time, operation).
    """
    mdl = cp_model.CpModel()

    # Make interval variables
    all_tasks = {}
    for op, t in tasks.items():
        st_var = mdl.NewIntVar(t.head, horizon, f'st_{op}')
        et_var = mdl.NewIntVar(t.head + t.p, horizon, f'et_{op}')
        inter_var = mdl.NewIntervalVar(st_var, t.p, et_var, f'inter_{op}')
        all_tasks[op] = CPTask(start=st_var, end=et_var, tail=t.tail,
                               interval=inter_var)

    # No overlap in time
    mdl.AddNoOverlap([task.interval for task in all_tasks.values()])

    # Force precedences among certain operation pairs
    if precs:
        print('\t\t--> Forcing precedences...')
        for op1, op2 in precs:
            mdl.Add(all_tasks[op2].start < all_tasks[op1].start)

    # Lmax objective
    lmax = mdl.NewIntVar(-horizon, horizon, 'Lmax')
    mdl.AddMaxEquality(lmax, [t.end - t.tail for t in all_tasks.values()])
    mdl.Minimize(lmax)

    # NOTICE: with more workers is not deterministic
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = args.seed
    solver.parameters.max_time_in_seconds = args.time_lmax
    solver.parameters.log_search_progress = DEBUG

    #
    if erd_init and precs is None:
        curr_st = 0
        for r, _, p, op in sorted([(t.head, t.tail, t.p, op)
                                   for op, t in tasks.items()]):
            op_st = max(curr_st, r)
            mdl.AddHint(all_tasks[op].start, op_st)
            curr_st += p

    status = solver.Solve(mdl)
    if status != cp_model.OPTIMAL:
        print(f'--> Lmax not OPTIMAL: {status}')

    # Retrieve the solution
    sol = sorted([(solver.Value(t.start), op) for op, t in all_tasks.items()])
    x = np.array([op for _, op in sol])
    return solver.ObjectiveValue(), x, sol


class ShiftingBottleneck(object):
    """
    Implementation of:
        Adams et al., The Shifting Bottleneck Procedure for Job Shop
        Scheduling, 1988

    NB: This implementation uses the ORTools for solving the lmax problem.
    """
    def __init__(self, max_reopt: int = 3):
        """
        Shifting Bottleneck Procedure

        Args:
            max_reopt: maximum number of reoptimization steps after the
                insertion of a new machine in the solution.
        """
        self.max_reopt = max_reopt

    def optimize(self, dag, tasks: dict):
        """
        Solve the Lmax problem on a single machine.
        Keep executing until obtained optimal permutation makes the partial
        solution feasible.

        Args:
            dag: directed acyclic graph of the partial solution under
                construction.
            tasks: list of tasks to be sequenced on the machine.
        Return:
            Machine permutation.
            Lmax value.
            Makespan of the partial solution with the added machine.
        """
        _ms = None
        precedences = []

        # Solve until feasible graph
        while _ms is None:
            # Solve the max lateness problem
            lmax, x, sol = solve_lmax(tasks, precedences)
            for src, dst in zip(x[:-1], x[1:]):
                dag.add_arc(src, dst)
            # Check feasibility
            _ms, cycle, dist = dag.makespan()
            if _ms is None:
                # If not feasible, remove the arcs and find the precedence
                # making a cycle in the graph
                not_found = True
                for src, dst in zip(x[:-1], x[1:]):
                    dag.remove_arc(src, dst)
                    if src in cycle and not_found:
                        not_found = False
                        n = cycle[src]
                        # if n not in x:
                        #     print(f'- Src: {src} -> Dst: {dst}')
                        #     print(f'- Cycle: {cycle}')
                        #     print(f'- X: {x}')
                        #     print(f'- N: {n}')
                        while n not in x:
                            # Next operation of src might not be on the machine
                            n = cycle[n]
                        while cycle[n] in x:
                            # Take last operation of the block
                            n = cycle[n]
                        assert src != n, f'Only op({n}) in cycle({cycle})?!'
                        precedences.append((src, n))
        return x, lmax, _ms

    def noncritical_reopt(self, dag: Graph, sol: np.ndarray, scheduled: list):
        """
        Reoptimize only machines with no arcs on the critical path.

        Args:
            dag: directed acyclic graph of the partial solution under
                construction.
            sol: 2d matrix of the solution.
            scheduled: sequence of machines already scheduled, necessary to use
                the right order.
        Return:
            The Directed Acyclic Graph after the reoptimization.
        """
        # Find critical machines
        best_ms, cp, dist = dag.makespan()
        assert best_ms, "Cycle in graph"
        critical_m = {dag.machines[op] for op in cp}

        # Remove non-critical machines
        num_remove = min(len(critical_m), int(np.sqrt(len(scheduled))))
        removed = []
        for m in scheduled[-1::-1]:
            if m not in critical_m:
                # Remove the machine
                removed.append(m)
                for src, dst in zip(sol[m][:-1], sol[m][1:]):
                    dag.remove_arc(src, dst)
                if len(removed) == num_remove:
                    break

        # Insert again non-critical machines
        costs = (dag.costs + EPS).astype(int)
        for m in removed[-1::-1]:
            perm = sol[m]
            # Compute release and due date
            _ms, _, long_to = dag.makespan()
            _, _, long_from = dag.makespan(reverse=True)
            tasks = {op: Task(p=costs[op],
                              head=int(long_to[op] - costs[op] + EPS),
                              tail=int(_ms - long_from[op] + costs[op] + EPS))
                     for op in perm}
            # Solve Lmax and save the machine permutation
            x, lmax, _ms = self.optimize(dag, tasks)
            sol[m] = x
        #
        print(f"\n\t\t- Noncritical reopt: {dag.makespan()[0]}")
        return dag

    def reoptimize(self, dag: Graph, sol: np.ndarray, scheduled: list):
        """
        Reoptimize the partial solution.
        Remove and reoptimize the machines in the solution for max_reopt
        cycles (default 3 as in Adams 1988).
        Finally, it also reoptimizes non-critical machines.

        Args:
            dag: directed acyclic graph of the partial solution under
                construction.
            sol: 2d matrix of the solution.
            scheduled: sequence of machines already scheduled, necessary to use
                the right order.
        Return:
            The Directed Acyclic Graph after the reoptimization.
        """
        costs = (dag.costs + EPS).astype(int)
        # Reoptimize until stop
        for i in range(self.max_reopt):
            scores = []
            for m in scheduled:
                # Remove the machine
                perm = sol[m]
                for src, dst in zip(perm[:-1], perm[1:]):
                    dag.remove_arc(src, dst)
                # Compute release and due date
                _ms, _, long_to = dag.makespan()
                _, _, long_from = dag.makespan(reverse=True)
                tasks = {op: Task(p=costs[op],
                                  head=int(long_to[op] - costs[op] + EPS),
                                  tail=int(_ms - long_from[op] + costs[op] + EPS))
                         for op in perm}
                # Solve Lmax and save the permutation
                new_perm, lmax, _ms = self.optimize(dag, tasks)
                scores.append((lmax, m))
                sol[m] = new_perm

            #
            scheduled = [x for _, x in sorted(scores, reverse=True)]
            print(f"\n\t\t- After reopt {i}: {dag.makespan()[0]}", flush=True)
        return self.noncritical_reopt(dag, sol, scheduled)

    def last_reoptimize(self, dag: Graph, sol: np.ndarray, scheduled: list,
                        max_last_reopt: int = 200, patient: int = 5):
        """
        Reoptimize the complete solution.
        Remove and reoptimize each machine from the solution until no
        improvement is observed in the makespan.

        Args:
            dag: directed acyclic graph of the partial solution under
                construction.
            sol: 2d matrix of the solution.
            scheduled: sequence of machines already scheduled, necessary to use
                the right order.
            max_last_reopt:
            patient:
        Return:
            The best Directed Acyclic Graph and its relative makespan.
        """
        costs = (dag.costs + EPS).astype(int)
        best_dag = dag.copy()
        best_ms = best_dag.makespan()[0]
        last_ms = float('inf')
        # Reoptimize the complete solution
        stop, cnt = False, patient
        while not stop and max_last_reopt:
            max_last_reopt -= 1
            scores = []
            stop = True
            for m in scheduled:
                # Remove the machine
                perm = sol[m]
                for src, dst in zip(perm[:-1], perm[1:]):
                    dag.remove_arc(src, dst)
                # Compute release and due date
                _ms, _, long_to = dag.makespan()
                _, _, long_from = dag.makespan(reverse=True)
                tasks = {op: Task(p=costs[op],
                                  head=int(long_to[op] - costs[op] + EPS),
                                  tail=int(_ms - long_from[op] + costs[op] + EPS))
                         for op in perm}
                # Solve and save the permutation
                new_perm, lmax, _ms = self.optimize(dag, tasks)
                scores.append((lmax, m))
                sol[m] = new_perm
                # Stop only if all the permutations are stable
                if abs(_ms - last_ms) > EPS:
                    stop = False
                # Save best solution produced
                if _ms < best_ms - EPS:
                    print(f"\t\t--> New best reopt: {_ms}")
                    best_ms = _ms
                    best_dag = dag.copy()
            # Wait 3 times before stopping
            if stop and cnt > 0:
                cnt -= 1
                stop = False
            else:
                cnt = patient
            # Recompute machine sequence
            scheduled = [m for _, m in sorted(scores, reverse=True)]
            last_ms = dag.makespan()[0]
            print(f"\n\t\t- Last reopt {max_last_reopt}: {last_ms}")
        return best_dag, best_ms

    def __call__(self, ins: dict):
        """
        Procedure entrypoint.

        Args:
            ins: Job Shop instance to solve.
        Return:
            Directed Acyclic Graph of the solution.
        """
        num_j, num_m = ins['j'], ins['m']
        machines = ins['machines'].reshape(-1)
        machine_ops = [np.argwhere(machines == m).squeeze()
                       for m in range(num_m)]
        #
        sol = -np.ones((num_m, num_j), dtype=np.int32)
        to_schedule = set(range(num_m))
        scheduled = []
        #
        dag = Graph(ins['costs'], ins['machines'], num_j, num_m)
        costs = (dag.costs + EPS).astype(int)
        print(f"\t- Init ms = {dag.makespan()[0]}")

        for i in range(num_m):
            print(f'\n############################################')
            print(f'\t- Iteration {i}:')
            max_lmax = -float('inf')
            bottleneck_perm, bottleneck_m = None, None

            it_ms, _, long_to = dag.makespan()
            _, _, long_from = dag.makespan(reverse=True)
            for m in tqdm(to_schedule):
                # Compute head and tail for each operation
                tasks = {op: Task(p=costs[op],
                                  head=int(long_to[op] - costs[op] + EPS),
                                  tail=int(it_ms - long_from[op] + costs[op] + EPS))
                         for op in machine_ops[m]}

                # Optimize the machine and add the permutation to the graph
                x, lmax, _ms = self.optimize(dag, tasks)
                # See "Scheduling" book of Pinedo for an explanation
                assert _ms >= it_ms + lmax, f"Error in Lmax (it={i}, m={m})"

                # Save the max lateness
                if lmax > max_lmax + EPS:
                    max_lmax = lmax
                    bottleneck_m, bottleneck_perm = m, x

                # Remove the added machine for the next iteration
                for src, dst in zip(x[:-1], x[1:]):
                    dag.remove_arc(src, dst)

            # Insert the permutation of the bottleneck machine
            sol[bottleneck_m] = bottleneck_perm
            for src, dst in zip(bottleneck_perm[:-1], bottleneck_perm[1:]):
                dag.add_arc(src, dst)
            print(f"\n\t\t- Before reopt: {dag.makespan()[0]}", flush=True)

            # Reschedule all the machines but the last added
            if 0 < i < num_m - 1:
                self.reoptimize(dag, sol, scheduled)
            to_schedule.remove(bottleneck_m)
            scheduled.append(bottleneck_m)
        #
        return self.last_reoptimize(dag, sol, scheduled,
                                    max_last_reopt=args.max_last_reopt)


if __name__ == '__main__':
    print(f'CONFIG: {args}')
    import pandas as pd
    from inout import load_dataset
    instances = load_dataset(f'benchmarks/{args.name}', basic=True)

    # 4 DEBUGGING
    # from inout import read_jsp
    # instances = [read_jsp(f'benchmarks/LA/la29.jsp')]
    # instances = [read_jsp(f'benchmarks/DMU/dmu75.jsp')]
    # instances = [read_jsp(f'benchmarks/TA/ta42.jsp')]

    #
    sbh = ShiftingBottleneck(max_reopt=args.max_reopt)
    for ins in instances:
        print(f'Solving {ins["name"]} ({ins["shape"]}): UB = {ins["makespan"]}')
        st = time()
        g, ms = sbh(ins)
        tt = time() - st

        # Check solution
        if ms is None:
            print(f"ERROR: {ins['name']}: unfeasible solution found")
        gap = (ms / ins['makespan'] - 1) * 100
        print(f"--> FINAL MS={ms} (GAP={gap:.3f})")

        # Save results
        df = pd.DataFrame([{'name': ins['name'],
                            'ub': ins['makespan'],
                            'time': tt,
                            'ms': ms,
                            'gap': gap}])
        df.to_csv(f'output/{args.name}_SBH.csv',
                  mode='a',
                  sep=',',
                  index=False,
                  header=False)
