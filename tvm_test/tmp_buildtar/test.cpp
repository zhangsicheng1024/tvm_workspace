#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <regex>

using namespace std;

void quickSort(vector<int> &v, int begin, int end) {
    if(begin >= end-1) return;
    int pivot = v[end-1];
    int i, j;
    for(i=0, j=0; j<end-1; j++) {
        if(v[j] < pivot) {
            int t = v[i];
            v[i] = v[j];
            v[j] = t;
            i++;
        }
    }
    v[end-1] = v[i];
    v[i] = pivot;
    quickSort(v, begin, i);
    quickSort(v, i+1, end);
}

int main() {
    vector<int> v = {1,3,5,2,4,6};
    for(int i : v) cout << i << " "; cout << endl;

    quickSort(v, 0, v.size());

    for(int i : v) cout << i << " "; cout << endl;

    return 0;
}