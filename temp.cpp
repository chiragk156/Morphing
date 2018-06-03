#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;

int main(int argc, char const *argv[])
{
	ifstream fp(argv[1]);
	int y,x;
	while(fp>>y>>x){
		cout<<x<<" "<<y<<endl;
	}
	return 0;
}