#include <stdio.h>
int main(){
    double x=1.0000001, y=2.0;
    for (long i=0;i<50000000L;i++){ x=x*1.0000001+0.9999999; if(x>1e6)x*=1e-6; }
    printf("likwid_mini: x=%f\n", x);
    return 0;
}
