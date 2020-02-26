#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "./common/CycleTimer.h"
#include "./common/graph.h"
#include "./common/grade.h"
#include "page_rank.h"

#define USE_BINARY_GRAPH 1
#define PageRankDampening 0.3f
#define PageRankConvergence 1e-7d

// used for check correctness
void reference_serial_pageRank(Graph g, double* solution, double damping,
                               double convergence) {
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    double* solution_new = new double[numNodes];
    double* score_old = solution;
    double* score_new = solution_new;
    bool converged = false;
    double broadcastScore = 0.0;
    double globalDiff = 0.0;
    int iter = 0;

    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }
    while (!converged && iter < MAXITER) {
        iter++;
        broadcastScore = 0.0;
        globalDiff = 0.0;
        for (int i = 0; i < numNodes; ++i) {
            score_new[i] = 0.0;

            if (outgoing_size(g, i) == 0) {
                broadcastScore += score_old[i];
            }
            const Vertex* in_begin = incoming_begin(g, i);
            const Vertex* in_end = incoming_end(g, i);
            for (const Vertex* v = in_begin; v < in_end; ++v) {
                score_new[i] += score_old[*v] / outgoing_size(g, *v);
            }
            score_new[i] = damping * score_new[i] + (1.0 - damping) * equal_prob;
        }
        for (int i = 0; i < numNodes; ++i) {
            score_new[i] += damping * broadcastScore * equal_prob;
            globalDiff += std::abs(score_new[i] - score_old[i]);
        }
        converged = (globalDiff < convergence);
        std::swap(score_new, score_old);
    }
    if (score_new != solution) {
        memcpy(solution, score_new, sizeof(double) * numNodes);
    }
    delete[] solution_new;
}

int main(int argc, char** argv) {

    int num_threads = -1;
    std::string graph_filename;

    if (argc < 3) {
        std::cerr << "Usage: <path/to/graph/file> <manual_set_thread_count>\n";
        exit(1);
    }

    int thread_count = -1;
    if (argc == 3) {
        thread_count = atoi(argv[2]);
    }
    if (thread_count <= 0) {
        std::cerr << "<manual_set_thread_count> must > 0\n";
        exit(1);
    }
    graph_filename = argv[1];

    Graph g;

    printf("----------------------------------------------------------\n");
    printf("Running with %d threads\n", thread_count);
    printf("----------------------------------------------------------\n");
    printf("Loading graph...\n");

    if (USE_BINARY_GRAPH) {
        g = load_graph_binary(graph_filename.c_str());
    } else {
        g = load_graph(argv[1]);
        printf("storing binary form of graph!\n");
        store_graph_binary(graph_filename.append(".bin").c_str(), g);
        delete g;
        exit(1);
    }
    printf("\n");
    printf("Graph stats:\n");
    printf("  Filename: %s\n", argv[1]);
    printf("  Edges: %d\n", g->num_edges);
    printf("  Nodes: %d\n", g->num_nodes);

    bool pr_check = true;
    double* sol1;
    sol1 = (double*) malloc(sizeof(double) * g->num_nodes);
    double* sol2;
    sol2 = (double*) malloc(sizeof(double) * g->num_nodes);

    double pagerank_base;
    double pagerank_time;

    double ref_pagerank_base;
    double ref_pagerank_time;

    double start;
    std::stringstream timing;
    std::stringstream ref_timing;

    timing << "Threads  Page Rank\n";
    ref_timing << "Serial Reference Page Rank\n";

    //Set thread count
    omp_set_num_threads(thread_count);

    //Run implementations
    start = CycleTimer::currentSeconds();
    pageRank(g, sol1, PageRankDampening, PageRankConvergence);
    pagerank_time = CycleTimer::currentSeconds() - start;

    //Run reference implementation
    start = CycleTimer::currentSeconds();
    reference_serial_pageRank(g, sol2, PageRankDampening, PageRankConvergence);
    ref_pagerank_time = CycleTimer::currentSeconds() - start;

    printf("----------------------------------------------------------\n");
    std::cout << "Testing Correctness of Page Rank\n";
    if (!compareApprox(g, sol2, sol1)) {
        pr_check = false;
    }

    if (!pr_check)
        std::cout << "Your Page Rank is not Correct" << std::endl;
    else
        std::cout << "Your Page Rank is Correct" << std::endl;

    char buf[1024];
    char ref_buf[1024];
    sprintf(buf, "%4d:   %.6f s\n",
            thread_count, pagerank_time);
    sprintf(ref_buf, "   1:   %.6f s\n",
            ref_pagerank_time);

    timing << buf;
    ref_timing << ref_buf;

    printf("----------------------------------------------------------\n");
    std::cout << "Serial Reference Summary" << std::endl;
    std::cout << ref_timing.str();

    printf("----------------------------------------------------------\n");
    std::cout << "Timing Summary" << std::endl;
    std::cout << timing.str();

    printf("----------------------------------------------------------\n");

    delete g;
    return 0;
}
