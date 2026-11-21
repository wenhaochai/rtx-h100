cmake_minimum_required(VERSION 3.24)
project(cycles_nrc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# tiny-cuda-nn: static, PIC, no RTC (no nvrtc dependency), no samples/benchmarks/bindings.
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC OFF CACHE BOOL "" FORCE)
if (NOT TCNN_CUDA_ARCHITECTURES)
  set(TCNN_CUDA_ARCHITECTURES "86;90" CACHE STRING "" FORCE)
endif()
add_subdirectory(third_party/tiny-cuda-nn)
# Patch patches/tcnn_nrc_minimal.patch (TCNN_NRC_MINIMAL): only the 3D 2-feature hash grid and the 64-neuron
# FullyFusedMLP are compiled, Permuto/LoD encodings are skipped; each file then builds in < 10 min on 7 cores.
set_source_files_properties(third_party/tiny-cuda-nn/src/encoding.cu third_party/tiny-cuda-nn/src/network.cu third_party/tiny-cuda-nn/src/fully_fused_mlp.cu DIRECTORY third_party/tiny-cuda-nn PROPERTIES COMPILE_DEFINITIONS TCNN_NRC_MINIMAL)

# Fused inference kernel (src/nrc_fused.cu): wgmma needs the arch-specific sm_90a target; mma.sync runs on sm_86.
# Only CuTe's GMMA atoms are used (header-only, CUTLASS 3.5.1 from tcnn).
set(NRC_CUTLASS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tiny-cuda-nn/dependencies/cutlass/include)
add_library(nrc_fused OBJECT src/nrc_fused.cu)
target_include_directories(nrc_fused PRIVATE ${NRC_CUTLASS_INCLUDE})
target_compile_options(nrc_fused PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
set_target_properties(nrc_fused PROPERTIES CUDA_ARCHITECTURES "86-real;90a-real")

# Frozen-weight inference (src/nrc_infer.cu, "engine": "infer"): hash or permutohedral lattice grid, MLP width 64-256,
# fp16 or fp8 wgmma. sm_90a only.
add_library(nrc_infer OBJECT src/nrc_infer.cu)
target_include_directories(nrc_infer PRIVATE ${NRC_CUTLASS_INCLUDE} third_party/tiny-cuda-nn/dependencies)
target_compile_options(nrc_infer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --expt-relaxed-constexpr>)
set_target_properties(nrc_infer PROPERTIES CUDA_ARCHITECTURES "90a-real")

add_library(cycles_nrc STATIC src/cycles_nrc.cu $<TARGET_OBJECTS:nrc_fused> $<TARGET_OBJECTS:nrc_infer>)
target_include_directories(cycles_nrc PUBLIC include)
target_link_libraries(cycles_nrc PUBLIC tiny-cuda-nn)
set_target_properties(cycles_nrc PROPERTIES CUDA_ARCHITECTURES "${TCNN_CUDA_ARCHITECTURES}")

# Test harness and benchmark (driver-API context like Cycles: primary context, non-blocking stream).
add_executable(nrc_test tests/nrc_test.cu)
target_link_libraries(nrc_test PRIVATE cycles_nrc cuda)
set_target_properties(nrc_test PROPERTIES CUDA_ARCHITECTURES "${TCNN_CUDA_ARCHITECTURES}")

# Host-only check of the wgmma fragment, swizzle and descriptor conventions against CuTe (tests/fused_layout_check.cu).
add_executable(fused_layout_check tests/fused_layout_check.cu)
target_include_directories(fused_layout_check PRIVATE ${NRC_CUTLASS_INCLUDE})
target_compile_options(fused_layout_check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(fused_layout_check PROPERTIES CUDA_ARCHITECTURES "86-real")
