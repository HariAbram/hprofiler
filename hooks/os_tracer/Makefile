# Builds the eBPF OS-level scheduler tracer (sched_trace.bpf.c) and its
# userspace loader (os_tracer.c) -- see both files' header comments and
# DOCUMENTATION.md's collection-path / os_tracer sections for the full
# design and verification status.
#
# Requires: clang (BPF backend), bpftool, and libbpf headers+library.
#
# On a properly set up system, install the dev package and this Makefile
# needs nothing further:
#   sudo apt install clang llvm libbpf-dev linux-tools-common linux-tools-$(uname -r)
#
# This was DEVELOPED on a machine with libbpf.so.1 (the runtime) but no
# libbpf-dev (the headers + unversioned .so symlink needed to compile/link
# against it) installed and no root access to install it. Worked around by
# borrowing the (unmodified, upstream-licensed) headers vendored inside
# the linux-headers package for building its own resolve_btfids tool, and
# linking directly against the versioned .so.1 instead of the usual -lbpf
# (which needs the unversioned symlink only -dev provides). LIBBPF_INC/
# LIBBPF_LIBS below default to that workaround so `make` reproduces the
# exact build verified in that environment; override them (or just
# install libbpf-dev, then LIBBPF_INC/LIBBPF_LIBS collapse to pkg-config's
# own answer) on any other machine.
KERNEL_REL := $(shell uname -r)
FALLBACK_LIBBPF_INC := /usr/src/linux-headers-$(KERNEL_REL)/tools/bpf/resolve_btfids/libbpf/include
LIBBPF_INC ?= $(shell pkg-config --cflags libbpf 2>/dev/null || echo -I$(FALLBACK_LIBBPF_INC))
LIBBPF_LIBS ?= $(shell pkg-config --libs libbpf 2>/dev/null || echo /usr/lib/x86_64-linux-gnu/libbpf.so.1 /usr/lib/x86_64-linux-gnu/libelf.so.1 /usr/lib/x86_64-linux-gnu/libz.so.1)

CC ?= gcc
CLANG ?= clang
BPFTOOL ?= bpftool

.PHONY: all clean
all: os_tracer

# CO-RE relies on this machine's own BTF-described kernel struct layouts --
# regenerated every build (not checked into source control -- it's
# kernel-version-specific, ~150k lines, and bpftool + a live kernel is all
# that's needed to produce it fresh).
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
