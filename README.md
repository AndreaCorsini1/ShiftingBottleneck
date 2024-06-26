# Shifting Bottleneck Heuristic

Python implementation of the Shifting Bottleneck Heuristic (Adams 1988) for the Job Shop Scheduling.
This implementation uses Cython to accelerate the computation of the makespan, and the CP-Sat of Google ORTools for solving the 
Lmax problem.

We also provide an implementation of the INSertion Algorithm (from "Fast Taboo Search", 
Nowicki and Smutnicki 1996) as a benchmark for comparison. 


## Installation:
Before running the code, you have to install the following packages:
- Networkx
- ORTools (9.8.3296)
- Cython
- Pandas
- tqdm

After having all set up, you have to build the Cython graph utils by running 
inside the cysource folder the following:
```
    python3 cython_setup.py build_ext --inplace
```


## Run with:

If you want to use the Shifting Bottleneck, run:
```
    python3 ShiftingBottleneck.py -name LA
```
Where "-name LA" is the argument to provide the name of the benchmark to test, located in the benchmarks folder.

While INSA can be used by running:
```
    python3 INSA.py -name LA
```

## Results & Times

A few results on Lawerence (LA) benchmark.

|      | Optimal |      INSA      | SBH (Adams) |   SBH (ours)   |
|:----:|:-------:|:--------------:|:-----------:|:--------------:|
| la01 |   666   | 666  (0.0 sec) |     666     | 666 (0.7 sec)  |
| la02 |   655   | 722  (0.0 sec) |     720     | 673 (0.8 sec)  |
| la03 |   597   | 685  (0.0 sec) |     623     | 622 (0.6 sec)  |
| la04 |   590   | 659  (0.0 sec) |     597     | 607 (0.6 sec)  |
| la05 |   593   | 593  (0.0 sec) |     593     | 593 (0.6 sec)  | 
| la06 |   926   | 932  (0.0 sec) |     926     | 926 (2.9 sec)  |
| la07 |   890   | 976  (0.0 sec) |     890     | 890 (4.3 sec)  |
| la08 |   863   | 868  (0.0 sec) |     868     | 863 (2.8 sec)  |
| la09 |   951   | 951  (0.0 sec) |     951     | 951 (4.2 sec)  |
| la10 |   958   | 958  (0.0 sec) |     959     | 958 (1.4 sec)  |
| la11 |  1222   | 1263 (0.0 sec) |    1222     | 1222 (5.2 sec) |
| la12 |  1039   | 1039 (0.0 sec) |    1039     | 1039 (3.5 sec) |
| la13 |  1150   | 1170 (0.0 sec) |    1150     | 1150 (9.6 sec) |
| la14 |  1292   | 1292 (0.0 sec) |    1292     | 1292 (7.6 sec) |
| la15 |  1207   | 1306 (0.0 sec) |    1207     | 1211 (6.4 sec) |

As the table shows, the results of our SBH version are aligned with those of Adams'
one.
As we coded the SBH in python and used CP-Sat for solving Lmax, the execution times 
are higher than an equivalent C version using the Carlier's B&B (as in the original paper). 
Note however that coding the B&B in Cython is also possible, if you plan to do that 
we can include it in the code.

You can find more results on Lawrence, Taillard, and Demirkol benchmarks in 
the output/Results.xlsx file.


## If you use it, cite us:
```
@inproceedings{SelfJSP,
    title = {Self-Labeling the Job Shop Scheduling Problem},
    author = {Corsini, Andrea and Porrello, Angelo and Calderara, Simone and Dell'Amico, Mauro},
    year={2024},
    publisher={Arxiv},
}
```