# The CPU MoE expert tier's shared library, and the bench/oracle that share its core.
#
# WHY THIS FILE EXISTS. `libcpumoe.so` is `.gitignore`d (it is a build artifact) and
# `weights/cpu_native.find_library` REFUSES rather than degrading when it is absent — there is no
# float64 fallback on purpose. So the build line has to live somewhere a person or an image build
# can run. Until 2026-09-07 it lived only in the text of that refusal message and in one agent's
# shell history, which is why the CPU tier had never once been reachable in a serve.
#
# The image builds this too (see the repo Dockerfile), so a turnkey `docker compose up` has it at
# /opt/kernels/libcpumoe.so. This target is for the dev path, where the repo is bind-mounted at
# /engine and `find_library` picks up /engine/tools/cpu_moe/libcpumoe.so.
#
# ARCH. `-march=znver4` is the box this was measured on (Ryzen 7 7800X3D, Zen 4) and it is what
# every number in RESULTS_*.txt was produced with. The core REQUIRES AVX-512 + VNNI + F16C; it does
# not have a scalar fallback, and it is not supposed to — the whole tier's premise is VPDPBUSD
# retiring 64 weights per instruction. Override for a different Zen: `make ARCH=znver5`.
ARCH ?= znver4
CXX ?= g++
CXXFLAGS ?= -O3 -march=$(ARCH) -mavx512vnni -mf16c -pthread -Wall

HDRS := moe_core.hpp wload.hpp

.PHONY: all so bench selftest clean
all: so

# THE SERVE ARTIFACT. Everything else here is measurement.
so: libcpumoe.so
libcpumoe.so: cpu_moe_so.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $<

# The checkpoint-driven bench/oracle. Same two headers, so a core change that breaks the bench
# breaks it HERE rather than after a 10-minute serve boot.
bench: cpu_moe_layer
cpu_moe_layer: cpu_moe_layer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ $<

# The intrinsics against their scalar-double twins, for every WLoad policy, in ~10 ms and with no
# checkpoint and no GPU. Run it after ANY edit to moe_core.hpp or wload.hpp.
selftest: cpu_moe_layer
	./cpu_moe_layer --mode selftest

clean:
	rm -f libcpumoe.so cpu_moe_layer
