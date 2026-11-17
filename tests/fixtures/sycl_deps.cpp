// SYCL dependencies across two queues of one device, for
// tests/integration/test_sycl_deps.py (AdaptiveCpp, generic SSCP target):
//
//   kA (q1) writes buffer a; kB (q2) reads it          -- an accessor dependency
//   kC (q1) writes USM u; kD (q2) depends_on(kC's event) -- an event dependency
//   a copy (q2) depends_on(kD's event)
//
// (AdaptiveCpp 25.10 has no host_task: its custom operations run at
// submission, not as dependent nodes.)
//
// argv[1] picks the device by backend: omp (the host device: the runtime
// orders these in its own worker threads), cuda or ocl (it lowers them to
// the backend's events / wait lists). "mixed": kA on the host device, kB on
// the CUDA device -- a buffer dependency across backends (only that one:
// the event dependency across backends hangs in AdaptiveCpp 25.10, with or
// without hooks). Prints "ok <backend>"
// when the results are right, "nodev <backend>" when there is no such
// device.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
  const char *want = argc > 1 ? argv[1] : "omp";
  bool mixed = !std::strcmp(want, "mixed");
  sycl::backend b = !std::strcmp(want, "cuda") || mixed ? sycl::backend::cuda
                    : !std::strcmp(want, "ocl")          ? sycl::backend::ocl
                                                         : sycl::backend::omp;
  sycl::device dev, host_dev;
  bool found = false, found_host = false;
  for (const auto &d : sycl::device::get_devices()) {
    if (!found && d.get_backend() == b) {
      dev = d;
      found = true;
    }
    if (!found_host && d.get_backend() == sycl::backend::omp) {
      host_dev = d;
      found_host = true;
    }
  }
  if (!found || (mixed && !found_host)) {
    std::printf("nodev %s\n", want);
    return 0;
  }

  constexpr size_t N = 1 << 18;
  constexpr int ITERS = 400;
  sycl::queue q1{mixed ? host_dev : dev}, q2{dev};
  sycl::buffer<float> a{sycl::range{N}}, out{sycl::range{N}};

  q1.submit([&](sycl::handler &h) {
    sycl::accessor w{a, h, sycl::write_only, sycl::no_init};
    h.parallel_for<class kA>(sycl::range{N}, [=](sycl::id<1> i) {
      float x = static_cast<float>(i[0] % 7);
      for (int k = 0; k < ITERS; ++k) x = x * 0.5f + 1.0f;
      w[i] = x;
    });
  });
  q2.submit([&](sycl::handler &h) {
    sycl::accessor r{a, h, sycl::read_only};
    sycl::accessor w{out, h, sycl::write_only, sycl::no_init};
    h.parallel_for<class kB>(sycl::range{N}, [=](sycl::id<1> i) { w[i] = r[i] * 2.0f; });
  });

  if (mixed) {   // a CUDA kernel depends_on a host-device event: hangs in AdaptiveCpp 25.10
    sycl::host_accessor ra{out};
    bool ok = ra[0] == 4.0f && ra[N - 1] == 4.0f;
    std::printf("%s %s\n", ok ? "ok" : "wrong", want);
    return ok ? 0 : 1;
  }
  float *u = sycl::malloc_device<float>(N, q2);
  float *host = sycl::malloc_host<float>(N, q2);
  sycl::event eC = q1.parallel_for<class kC>(sycl::range{N}, [=](sycl::id<1> i) {
    float x = 1.0f;
    for (int k = 0; k < ITERS; ++k) x = x * 0.5f + 1.0f;
    u[i] = x;
  });
  sycl::event eD = q2.submit([&](sycl::handler &h) {
    h.depends_on(eC);
    h.parallel_for<class kD>(sycl::range{N}, [=](sycl::id<1> i) { u[i] += 1.0f; });
  });
  q2.submit([&](sycl::handler &h) {
    h.depends_on(eD);
    h.memcpy(host, u, N * sizeof(float));
  });
  q1.wait();
  q2.wait();
  bool host_ok = host[0] == 3.0f && host[N - 1] == 3.0f;

  sycl::host_accessor ra{out};
  bool ok = host_ok && ra[0] == 4.0f && ra[N - 1] == 4.0f;     // x converges to 2
  sycl::free(u, q2);
  sycl::free(host, q2);
  std::printf("%s %s\n", ok ? "ok" : "wrong", want);
  return ok ? 0 : 1;
}
