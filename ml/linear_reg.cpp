// lin reg in c++

#include<bits/stdc++.h>
using namespace std;
#define fin(i,j,k) for(int i=j;i<k;i++)

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
	double flt() {
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

mat<double> rnd_mat(int _rows, int _cols, double lo, double hi) {
	mat<double> res(_rows, _cols);
	fin(i,0,_rows) {
		fin(j,0,_cols) {
			res[i][j]=lo+(hi-lo)*rng.flt();
		}
	}
	return res;
}

mat<double> cnst_mat(int _rows, int _cols, double val) {
	mat<double> res(_rows, _cols);
	fin(i,0,_rows) {
		fin(j,0,_cols) {
			res[i][j]=val;
		}
	}
	return res;
}

mat<double> transpose(mat<double> M) {
	mat<double> res(M.cols, M.rows);
	fin(i,0,M.rows) {
		fin(j,0,M.cols) {
			res[j][i]=M[i][j];
		}
	}
	return res;
}

double mat_sum(mat<double> M) {
	double res=0;
	fin(i,0,M.rows) {
		fin(j,0,M.cols) {
			res+=M[i][j];
		}
	}
	return res;
}

double mse(mat<double> Y_, mat<double> Y) {
	double res = 0;
	fin(i,0,Y.rows) {
		fin(j,0,Y.cols) {
			res+=(Y_[i][j]-Y[i][j])*(Y_[i][j]-Y[i][j]);
		}
	}
	return res/Y.rows;
}

struct linreg {
	double lr=0.01;
	int iter=1000;
	mat<double> weights;
	mat<double> bias;

	linreg(double _lr, int _iter) {
		lr=_lr;
		iter=_iter;
	}
	linreg() {}

	void fit(mat<double> X, mat<double> Y) {
		int n_samples = X.rows;
		int n_features = X.cols;
		int y_features = Y.cols;
		weights = rnd_mat(n_features,y_features,0,1);
		bias = cnst_mat(n_samples, y_features, rng.flt()*10);
		fin(_,0,iter) {
			mat<double> Y_pred = X*weights + bias;
			mat<double> dw = transpose(X)*(Y_pred - Y);
			dw = dw * (1.0/n_samples);
			mat<double> db = cnst_mat(bias.rows, bias.cols, mat_sum((Y_pred-Y)));
			db = db * (1.0/n_samples);
			
			weights = weights - (dw * lr);
			bias = bias - (db * lr);
			if(_%5000==0) {
				cout<<"mse: "<<_/5000<<" : "<<mse(Y_pred, Y)<<endl;
			}
			// weights.print();
			// bias.print();
		}
	}

	mat<double> predict(mat<double> X) {
		return (X*weights + bias);
	}
};

int main() {
	int samples, x_ft, y_ft;
	cin>>samples>>x_ft>>y_ft;
	mat<double> X=rnd_mat(samples, x_ft, 0, 10);
	mat<double> Y=rnd_mat(samples, y_ft, 0, 10);
	// X.print();
	// Y.print();
	linreg test(0.001, 100000);
	test.fit(X, Y);

	mat<double> res=test.predict(X);
	Y.print();
	res.print();
}