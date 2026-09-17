#include <cstdio>
#include <vector>
#include <omp.h>
int main() {
    std::vector<double> v(2000000, 1.0);
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v.size(); i++) sum += v[i] * 1.0001;
#pragma omp parallel
#pragma omp single
    {
#pragma omp taskloop grainsize(2048)
        for (size_t i = 0; i < v.size(); i++) v[i] *= 1.0001;
    }
    printf("omp_mini: sum=%f threads=%d\n", sum, omp_get_max_threads());
    return 0;
}
