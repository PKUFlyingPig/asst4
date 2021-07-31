#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  
  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
    Vertex* sink_nodes = new int[numNodes];
    int sink_nodes_num = 0;
    for (Vertex i = 0; i < numNodes; i++) {
      if (outgoing_size(g, i) == 0) {
          sink_nodes[sink_nodes_num++] = i;
      }
    }
    bool converged = false;
    double rest = (1.0 - damping) / numNodes;
    double *score_new = new double[numNodes];
    while (!converged) {
      double randomJumpPr = 0.0;
      #pragma omp parallel for reduction (+:randomJumpPr) schedule(dynamic,200)
      for (int j = 0; j < sink_nodes_num; j++) {
        randomJumpPr += solution[sink_nodes[j]];
      } 
      randomJumpPr *= damping;
      randomJumpPr /= numNodes;

      #pragma omp parallel for schedule(dynamic,200)
      for (int i=0; i<numNodes; i++) {
        // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
        score_new[i] = 0;
        const Vertex* start = incoming_begin(g, i);
        const Vertex* end = incoming_end(g, i);
        for (const Vertex* v=start; v!=end; v++) {
          int vv = *v;
          score_new[i] += solution[vv] / outgoing_size(g, vv);
        }
        score_new[i] = (damping * score_new[i]) + rest + randomJumpPr;
      }

      double global_diff = 0;
      #pragma omp parallel for reduction (+:global_diff) schedule(dynamic,200)
      for (int i = 0; i < numNodes; i++) {
        global_diff += abs(score_new[i] - solution[i]);
        solution[i] = score_new[i];
      }
      converged = (global_diff < convergence);
    }
    free(score_new);
}
