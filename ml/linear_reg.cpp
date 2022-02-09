// lin reg in c++

#include<bits/stdc++.h>
using namespace std;
#define fin(i,j,k) for(int i=j;i<k;i++)
#define bigflt long double

struct prng{
private:
	unsigned long long state=1;
	void next_rand() {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
	}
public:
	prng(unsigned long long _state) {
		state=_state;
	}
	bigflt flt() {
		next_rand();
		unsigned long long cur = state;
		return cur*1.0/ULLONG_MAX;
	}
};
prng rng(time(0));

template<typename T>
struct mat{
	int rows=0;
    int cols=0;
	vector<vector<T>> v;
	mat(int _rows,int _cols) {
		rows = _rows;
        cols = _cols;
		v=vector<vector<T>>(rows,vector<T>(cols,0));
	}
	mat() {}
    vector<T> &operator [](int i) {
        return v[i];
    }
    mat operator *(mat other) {
        // cur * other
        // cur.cols == other.rows
        mat res(rows,other.cols);
        fin(i,0,rows) 
            fin(j,0,other.cols) 
                fin(k,0,cols) 
                    res[i][j]=(res[i][j]+v[i][k]*other[k][j]);
        return res;
    }
    mat operator *(T other) {
        mat res(rows, cols);
        fin(i,0,rows)
            fin(j,0,cols)
                res[i][j]=v[i][j]*other;
        return res;
    }
    mat operator +(mat other) {
        mat<T> res(rows,cols);
        fin(i,0,rows)
            fin(j,0,cols)
                res[i][j]=v[i][j]+other[i][j];
        return res;
    } 
    mat operator -(mat other) {
    	mat<T> res(rows, cols);
    	fin(i,0,rows)
    		fin(j,0,cols)
    			res[i][j]=v[i][j]-other[i][j];
    	return res;
    }
    void print() const{
        fin(i,0,rows) {
            for(auto el:v[i])
            	cout<<el<<" ";
            cout<<"\n";
        }
        cout<<"\n";
    }
};

mat<bigflt> rnd_mat(int _rows, int _cols, bigflt lo, bigflt hi) {
	mat<bigflt> res(_rows, _cols);
	fin(i,0,_rows) {
		fin(j,0,_cols) {
			res[i][j]=lo+(hi-lo)*rng.flt();
		}
	}
	return res;
}

mat<bigflt> cnst_mat(int _rows, int _cols, bigflt val) {
	mat<bigflt> res(_rows, _cols);
	fin(i,0,_rows) {
		fin(j,0,_cols) {
			res[i][j]=val;
		}
	}
	return res;
}

mat<bigflt> transpose(mat<bigflt> M) {
	mat<bigflt> res(M.cols, M.rows);
	fin(i,0,M.rows) {
		fin(j,0,M.cols) {
			res[j][i]=M[i][j];
		}
	}
	return res;
}

bigflt mat_sum(mat<bigflt> M) {
	bigflt res=0;
	fin(i,0,M.rows) {
		fin(j,0,M.cols) {
			res+=M[i][j];
		}
	}
	return res;
}

bigflt mse(mat<bigflt> Y_, mat<bigflt> Y) {
	bigflt res = 0;
	fin(i,0,Y.rows) {
		fin(j,0,Y.cols) {
			res+=(Y_[i][j]-Y[i][j])*(Y_[i][j]-Y[i][j]);
		}
	}
	return res/Y.rows;
}

struct linreg {
	bigflt lr=0.01;
	int iter=1000;
	mat<bigflt> weights;
	mat<bigflt> bias;
	mat<bigflt> X;
	mat<bigflt> Y;
	int n_samples = 0;
	int n_features = 0;
	int y_features = 0;
	linreg(mat<bigflt> _X, mat<bigflt> _Y) {
		X = _X;
		Y = _Y;
		weights = rnd_mat(X.cols, Y.cols,0,1);
		bias = cnst_mat(X.rows, Y.cols, rng.flt()*10);
		n_samples = X.rows;
		n_features = X.cols;
		y_features = Y.cols;
	}
	void fit(bigflt _lr, int _iter) {
		lr = _lr;
		iter = _iter;
		fin(_,0,iter) {
			mat<bigflt> Y_pred = X*weights + bias;
			mat<bigflt> dw = transpose(X)*(Y_pred - Y);
			dw = dw * (1.0/n_samples);
			mat<bigflt> db = cnst_mat(bias.rows, bias.cols, mat_sum((Y_pred-Y)));
			db = db * (1.0/n_samples);
			
			weights = weights - (dw * lr);
			bias = bias - (db * lr);
			if(_%500==0) {
				cout<<"mse: "<<_/500<<" : "<<mse(Y_pred, Y)<<endl;
			}
			// weights.print();
			// bias.print();
		}
	}

	void train(bigflt initlr, int epochs, int epochsize) {
		while(epochs--) {
			fit(initlr, epochsize);
			initlr/=10;
		}
	}

	mat<bigflt> predict(mat<bigflt> X) {
		return (X*weights + bias);
	}
};

int main() {
	int samples, x_ft, y_ft;
	cin>>samples>>x_ft>>y_ft;
	mat<bigflt> X=rnd_mat(samples, x_ft, 0, 10);
	mat<bigflt> Y=rnd_mat(samples, y_ft, 0, 10);
	// X.print();
	// Y.print();
	linreg test(X,Y);
	test.train(0.01, 10, 1000);
	// bigflt lr = 0.01;
	// fin(i,0,10) {
	// 	test.fit(lr, 1000);
	// 	lr/=10;
	// }
	// test.fit(0.01, 1000);
	// test.fit(0.01, 1000);
	// test.fit(0.001, 10000);
	// test.fit(0.0001, 1000);

	mat<bigflt> res=test.predict(X);
	cout<<"actual values: "<<endl;
	Y.print();
	cout<<"pred values: "<<endl;
	res.print();
	cout<<"error : "<<endl;
	cout<<mse(Y,res)<<endl;
}