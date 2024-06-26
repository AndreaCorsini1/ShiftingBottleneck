from argparse import ArgumentParser

# Tolerance for float comparison
EPS = 1e-3
#
DEBUG = False


parser = ArgumentParser(description='Configuration for heuristics')
parser.add_argument("-name", type=str, required=False,
                    default='TA',
                    help="Name of the benchmark in benchmarks folder.")
parser.add_argument("-seed", type=int, required=False,
                    default=12345,
                    help="Random seed for CP-Sat.")
##### Shifting Bottleneck Heuristic
parser.add_argument("-max_reopt", type=int, default=3,
                    required=False,
                    help="Number of max reopts after inserting a machine.")
parser.add_argument("-max_last_reopt", type=int, default=200,
                    required=False,
                    help="Number of max reopts in the final step.")
parser.add_argument("-time_lmax", type=int, default=600,
                    required=False,
                    help="Time limit in seconds for solving Lmax.")
args = parser.parse_args()
