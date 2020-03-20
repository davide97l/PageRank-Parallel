#include <iostream>
#include <fstream>
#include <list>
#include <iterator>
#include <map>
#include <omp.h>
#include <time.h>

using namespace std;

struct Page {
    list <int> ids_in; //in-pages
    int num_ids_in; //number of in-pages
    int num_ids_out; ////number of out-pages
    float score; //page-rank score
    float score_new; //used as temp variable to update the score
    Page(){
        num_ids_in = 0;
        num_ids_out = 0;
        score = 0.0;
    }
};

//function for printing the elements in a list of int
void showListInt(list <int> g){
    for(list <int> :: iterator it = g.begin(); it != g.end(); ++it)
        cout<<*it<<" ";
}
//show map of pages (full details)
void showPagesFull(map <int, Page> g){
    for(auto& x : g){
        std::cout << x.first<<" "<< x.second.num_ids_in<<" "<<x.second.num_ids_out<<" "<<x.second.score <<" ";
        showListInt(x.second.ids_in);
        cout<<"\n";
    }
}
//show map of pages (only score)
void showPages(map <int, Page> g){
    for(auto& x : g){
        std::cout << x.first<<"\t"<<x.second.score <<"\n";
    }
}

#define iterations 5
#define damping 0.85
#define n_threads 8
#define input_file "linkgraph.txt"

int main(int argc, char** argv){
    //Set thread count
    omp_set_num_threads(n_threads);
    ifstream cin(input_file);
    int num_pages = 0;
    int id, out;
    map <int, Page> pages;
    map <int, int> lookup;
    cout<<"Loading graph data..."<<endl;
    clock_t tStart = clock();
    while (cin>>id){
        cin>>out;
        if(!pages.count(id)){
            pages[id] = Page();
            lookup[num_pages]=id;
            num_pages++;
        }
        if(!pages.count(out)){
            pages[out] = Page();
            lookup[num_pages]=out;
            num_pages++;
        }
        pages[out].num_ids_in++;
        pages[out].ids_in.push_back(id);
        pages[id].num_ids_out++;
    }
    cout<<"Graph data loaded in "<<(double)(clock() - tStart)/CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"Loaded "<<num_pages<<" pages"<<endl;

    // initialize pageRank
    float equal_prob = 1.0 / num_pages;
    for(auto& x : pages)
        x.second.score = equal_prob;

    float broadcastScore;

    cout<<"Begin algorithm..."<<endl;
    clock_t aStart = clock();
    for(int j=0;j<iterations;j++){
        clock_t tStart = clock();
        broadcastScore = 0.0;

        // evaluate the score for each node
        #pragma omp parallel for reduction(+: broadcastScore) schedule (dynamic, 32)
        for(int i=0;i<pages.size();i++){

            // find the ID corresponding to the i-page
            int idx = lookup[i];

            // used to store the new score
            pages[idx].score_new = 0.0;

            // if the node has no outgoing edges, then add its value to the broadcast score
            if(!pages[idx].num_ids_out)
                broadcastScore += pages[idx].score;

            // iterate over all the vertices with an incoming edge to compute the new value of the vertices
            if(pages[idx].num_ids_in>0){
                for(list <int> :: iterator it = pages[idx].ids_in.begin(); it != pages[idx].ids_in.end(); ++it)
                    pages[idx].score_new += pages[*it].score / pages[*it].num_ids_out;
            }

            // apply pageRank equation
            pages[idx].score_new = damping * pages[idx].score_new + (1.0 - damping) / num_pages;
        }

        // update broadcast score
        broadcastScore = damping * broadcastScore / num_pages;
        for(auto& x : pages){

            // add the global broadcast score to each edge to compute its final score
            x.second.score = x.second.score_new + broadcastScore;
        }
        cout<<"Iteration "<<j<<" completed in "<<(double)(clock() - tStart)/CLOCKS_PER_SEC<<"s"<<endl;
    }
    cout<<"Algorithm terminated in "<<(double)(clock() - tStart)/CLOCKS_PER_SEC<<"s"<<endl;

    clock_t rStart = clock();
    cout<<"Writing result..."<<endl;
    string result_path = string("output_") + string(input_file);
    ofstream result_file(result_path);
    for(auto& x : pages){
        result_file << x.first<<"\t"<<x.second.score<<"\n";
    }
    cout<<"Results written in "<<(double)(clock() - rStart)/CLOCKS_PER_SEC<<"s"<<endl;
    result_file.close();

    //showPages(pages);
}
