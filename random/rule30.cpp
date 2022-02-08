// automation: some transformation of state
// state[i] ---> some func of state[i-1]
// in this example, state[i] is 
// only dependant on neighbouring i-1 states
// state[i][j] = f(state[i-1][j], state[i-1][j], state[i-1][j+1])
// rule 30 : maps to 00011110 (binary rep of 30)
// https://en.wikipedia.org/wiki/Rule_30

#include<bits/stdc++.h>
using namespace std;

struct automation {
	int size=0;
	int rule=0;
	vector<bool> state;
	automation(int _sz, int _rl, vector<bool> _st) {
		state=_st;
		size=_sz;
		rule=_rl;	
	}
	void next() {
		vector<bool> _next(size);
		for(int i=0;i<size;i++) {
			int _cur=(i>0?state[i-1]:0)*4+state[i]*2+(i<size-1?state[i+1]:0);
			_next[i]=rule & (1<<(_cur));
		}
		state=_next;
	}
	void print() {
		for(int i=0;i<size;i++)
			cout<<(state[i]?'#':'.');
		cout<<endl;
	}
};


int main() {
	int n,rule;
	cin>>n>>rule;
	vector<bool> init(n,false);
	init[n/2]=true;
	// initial state : .........#.........
	automation r30(n,rule,init);
	for(int i=0;i<n/2+1;i++) {
		r30.print();
		r30.next();
	}
}