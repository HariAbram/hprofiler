# Builds the eBPF scheduler tracer (sched_trace.bpf.c) and its userspace
# loader (os_tracer.c). Requires clang (BPF backend), bpftool, and libbpf
# headers + library:
#   sudo apt install clang llvm libbpf-dev linux-tools-common linux-tools-$(uname -r)
#
# With libbpf-dev installed, LIBBPF_INC/LIBBPF_LIBS come from pkg-config.
# The fallback defaults below build without it, using the unmodified libbpf
# headers vendored in the linux-headers package and the versioned
# libbpf.so.1 (the unversioned symlink -lbpf needs ships only in -dev).
KERNEL_REL := $(shell uname -r)
FALLBACK_LIBBPF_INC := /usr/src/linux-headers-$(KERNEL_REL)/tools/bpf/resolve_btfids/libbpf/include
LIBBPF_INC ?= $(shell pkg-config --cflags libbpf 2>/dev/null || echo -I$(FALLBACK_LIBBPF_INC))
LIBBPF_LIBS ?= $(shell pkg-config --libs libbpf 2>/dev/null || echo /usr/lib/x86_64-linux-gnu/libbpf.so.1 /usr/lib/x86_64-linux-gnu/libelf.so.1 /usr/lib/x86_64-linux-gnu/libz.so.1)

CC ?= gcc
CLANG ?= clang
BPFTOOL ?= bpftool

.PHONY: all clean
all: os_tracer

# CO-RE layouts come from the running kernel's BTF; vmlinux.h is generated
# on every build (kernel-specific, ~150k lines, not in source control).
vmlinux.h:
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

sched_trace.bpf.o: sched_trace.bpf.c vmlinux.h
	$(CLANG) -target bpf -D__TARGET_ARCH_x86 -g -O2 -I. $(LIBBPF_INC) -c $< -o $@

sched_trace.skel.h: sched_trace.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

os_tracer.o: os_tracer.c sched_trace.skel.h
	$(CC) -Wall -Wextra -O2 $(LIBBPF_INC) -c os_tracer.c -o $@

os_tracer: os_tracer.o
	$(CC) -o $@ $< $(LIBBPF_LIBS)

clean:
	rm -f vmlinux.h sched_trace.bpf.o sched_trace.skel.h os_tracer.o os_tracer
