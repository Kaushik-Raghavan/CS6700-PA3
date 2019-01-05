#include <bits/stdc++.h>

using namespace std;

#define PB push_back
#define MP make_pair
#define X first
#define Y second

#define cil(a,b) ( ((a)%(b) == 0)?((a)/(b)):((a)/(b)+1) )
#define err(x) cerr << #x << " = " << x << endl;
#define forall(i,a,b) for(int i=a;i<b;i++)
#define foreach(v, c) for(typeof((c).begin())v=(c).begin();v!=(c).end();++v)
#define read(a) freopen(a,"r",stdin)
#define write(a) freopen(a,"w",stdout)

typedef long long ll;
typedef long long l;
typedef double db;
typedef vector<int> vi;
typedef vector<long long> vll;
typedef pair<int,int> pii;
typedef pair<ll,ll> pll;
typedef pair<double,double> pdd;
typedef vector<pii> vii;

clock_t start;
void Time(bool timeIt) {
	if (!timeIt) return;
	clock_t end = clock();
	double elapsed_time = ((db)end - (db)start) / (db)CLOCKS_PER_SEC;
	fprintf(stderr, "Time elapsed = %0.4lf\n", elapsed_time); 
}

#define LINF (long long)1e18
#define EPS 1e-9
#define INF 1000000007ll
#define SIZE 100010
#define MAX_A 1000010

int main() {
	start = clock();

	for (int i = 0 ; i < 13 ; ++i) cout << "6 ";
	cout << endl;
	for (int i = 1 ; i < 12 ; ++i) {
		cout << "6 ";
		for (int j = 1 ; j < 12 ; ++j) {
			cout << "0 ";
		}
		cout << "6" << endl;
	}
	for (int i = 0 ; i < 13 ; ++i) cout << "6 ";
	cout << endl;

	Time(true);
	return 0;
}
