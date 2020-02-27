# PageRank-Parallel

# DCGAN

## How to use

First of all use the make command to compile the files.
```
$ make
```
Afterwards, run the following command to launch the program your graph using a certain number of threads.
```
$ ./pr <path to graph> <number of threads> 
```
For example, run the following command to launch the program on the toy graph using 16 threads.
```
$ ./pr /tools/binary_simple.graph 16 
```
If you want to change the chunk size or the scheduling policy you will have to do it by modifiend the following line in the `page_rank.cpp` file. Replace `dynamic` with another policy and 16 with your `chunk` size.
```
$ #pragma omp parallel for reduction(+: broadcastScore) schedule(dynamic, 16)
```
Sometimes you may want to make their own graphs for debugging. You can write down a graph definition in a text file and use the graphTools app to convert it to a binary file. See the command help for: 
```
$ ./tools/graphTools text2bin <graphtextfilename> <graphbinfilename>
```

## Project report

- https://davideliu.com/2020/02/27/analysis-of-parallel-version-of-pagerank-algorithm/

