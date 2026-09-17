#include <stdio.h>
typedef int hipError_t;
extern hipError_t hipGetDeviceCount(int*);
extern hipError_t hipMalloc(void**, unsigned long);
extern const char* hipGetErrorString(hipError_t);
int main(){
    int n=-1; hipGetDeviceCount(&n);
    void *p=0; hipError_t e = hipMalloc(&p, 1024);
    printf("hip_mini: devcount=%d malloc=%s\n", n, hipGetErrorString(e));
    return 0;
}
