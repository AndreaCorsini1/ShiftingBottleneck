import os
import numpy as np


def read_jsp(f_path: str):
    """
    Load a Job Shop instance.

    Args:
        f_path: Read the basic info of a JSP instance.
    """
    with open(f_path) as f:
        #
        shape = next(f).split()
        n = int(shape[0])
        m = int(shape[1])
        name = f_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

        # Load the instance
        instance = -np.ones((n, 2 * m), dtype=np.int32)
        for j in range(n):
            instance[j] = np.array([int(d) for d in f.readline().split()])
        assert (instance >= 0).all(), f"Missing information {f_path}!"

        # Load the UB
        ms = 0
        try:
            _ms = next(f)
            if _ms != '':
                ms = int(_ms)
        except StopIteration:
            pass
    return {
        'name': name, 'j': n, 'm': m, 'path': f_path,
        'shape': f"{n}x{m}",
        'machines': instance[:, ::2],
        'costs': instance[:, 1::2],
        'makespan': ms,
    }


def load_dataset(path: str = './benchmarks/TA', **kwargs):
    """
    Load the dataset.

    Args:
        path: Path to the folder that contains the instances.
    Returns:
        instances: (list)
    """
    print(f"Loading {path} ...")
    #
    instances = []
    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        #
        instances.append(read_jsp(os.path.join(path, file)))
    #
    print(f"\tNumber of instances = {len(instances)}")
    return instances
