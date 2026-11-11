# Strata Engine - top-level build.
#
# One project builds the engine for every backend: CUDA (STRATA_ENABLE_CUDA), HIP (STRATA_ENABLE_HIP; the layer-split
# and DeepSeek engines are the STRATA_BUILD_SPLIT / STRATA_BUILD_DSV41 options), the CPU-only build, and the test
# programs.  The Intel Arc (SYCL) engine is a separate project in sycl/ (STRATA_ENABLE_SYCL), and the Metal engine
# one in metal/.  CMakePresets.json has the CPU check preset (`cmake --preset cpu-check`) and the compile-only
# backend presets used by tools/check and the CI workflows; the release scripts in packaging/ pass their own flags.
#
# The CUDA kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard below refuses anything older: silently
# building for a pre-Ampere arch would produce a binary that runs and is wrong about its own performance, which is
# the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.59.1 LANGUAGES CXX)
# MSVC on a non-UTF-8 code page (949 on a Korean Windows) warns C4819 for every non-ASCII byte in CUDA's own headers:
# over 12,000 lines in one strata-poly build.  /wd4819 only silences that warning (the code compiles as before, unlike
# /utf-8); it reaches ggml too, and the .cu files through -Xcompiler.
add_compile_options($<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:MSVC>>:/wd4819>
                    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4819>)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_geometry.cmake)   # -DSTRATA_GEOMETRY=<name>: the model shape the kernels are compiled for (default flash_next, no flag added)
# The build is cut by backend (cleanup 5.4): each file below only DEFINES macros (strata_<part>()), which run here, in this scope, at the place their
# block used to be - so this file reads in the order the old single CMakeLists.txt evaluated.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_ggml.cmake)          # which ggml: the vendored third_party/strata-ggml, or -DSTRATA_GGML_DIR
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/cpu.cmake)         # strata_kernels_cpu, the native (ggml-cpu) expert rows, the CPU tests
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/cuda.cmake)        # NVIDIA: toolchain, llama.cpp's MMQ kernels (strata_mmq, strata_kq)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/hip.cmake)         # AMD: gfx906 compat, MMQ on HIP
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_gpu_engine.cmake)    # the GPU engine both of them build
# EXPERIMENTAL: the Intel Arc engine, the SYCL port in sycl/ (PR #423; icpx + oneMKL, docs/INTEL_ARC.md). It is its
# own project that takes the unported sources from src/; with this on, nothing below is configured.
option(STRATA_ENABLE_SYCL "EXPERIMENTAL: build the Intel Arc (SYCL) engine from sycl/ - docs/INTEL_ARC.md" OFF)
if(STRATA_ENABLE_SYCL)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_ENABLE_SYCL is a separate backend; leave STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")
  endif()
  message(WARNING "STRATA_ENABLE_SYCL: the Intel Arc engine is experimental (community port, docs/INTEL_ARC.md)")
  add_subdirectory(sycl)
  return()
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Strata's own AVX2/AVX-512 kernels stay in the
# binary behind their CPU checks; the router lookahead gets an AVX1 kernel.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# #3202: Windows hands a program its command line in the system's ANSI code page (cp1251 / cp866 on a Russian Windows, cp1252 on an
# English one), but the engine reads every path as UTF-8 (the run configs and the server write UTF-8). A model folder with
# Cyrillic letters arrived as other bytes, or as "?" where the page has no such letter, and the model did not open. This manifest
# makes the process's ANSI code page UTF-8 (Windows 10 1903 and newer; older Windows ignores it and nothing changes), so the
# command line, fopen and std::ifstream all take UTF-8.
function(strata_utf8_manifest target)
  if(MSVC)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/strata-utf8.manifest)
  endif()
endfunction()

# The AVX kernels exist for x86 only.  On aarch64 (DGX Spark / GB10) portable.cpp stands in for them and
# native experts run on ggml-cpu's NEON dot products.
if(NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64|i[3-6]86|x86|X86)$")   # an unset processor stays the x86 build, as before
  set(STRATA_CPU_X86 ON)
else()
  set(STRATA_CPU_X86 OFF)
endif()
if(STRATA_ISA_FLOOR AND NOT STRATA_CPU_X86)
  message(WARNING "STRATA_ISA_FLOOR is an x86 older-CPU build and does not apply to ${CMAKE_SYSTEM_PROCESSOR}; ignored")
  set(STRATA_ISA_FLOOR "" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
# AMD gfx906 (MI50, wave64): the same .cu sources compiled as HIP through the gfx906 compat layer.  The CUDA runtime and
# cuBLAS names are mapped by the headers in include/strata/platform/hip_compat/, which go first on the include path,
# and CUDA::cudart / CUDA::cublas become aliases of HIP's targets, so every CUDA target below stays as it is.  This is
# separate from STRATA_ENABLE_HIP (the wave32 RDNA backend): with it on, STRATA_ENABLE_CUDA is on and
# STRATA_ENABLE_HIP stays off, so every "CUDA" branch of the tree is the one taken.
option(STRATA_HIP_GFX906 "Build the CUDA targets for AMD gfx906 through HIP (wave64 compat layer)" OFF)
if(STRATA_HIP_GFX906)
  if(STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_HIP_GFX906 and STRATA_ENABLE_HIP are separate backends; enable only one")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
# The community-made AMD changes (include/strata/core/amd_community.hpp): on by default.  This option is the build half of the one
# switch STRATA_AMD_COMMUNITY_TUNING: OFF compiles the portable signed-byte dot for gfx1010 again (the SDWA sequence of #4855).
# The rocBLAS tables are switched at run time (STRATA_AMD_COMMUNITY_TUNING=0).
option(STRATA_AMD_COMMUNITY_TUNING "Compile the community-made gfx1010 SDWA dot product (the runtime switch of the same name controls the gfx1030 rocBLAS tables)" ON)
if(NOT STRATA_AMD_COMMUNITY_TUNING)
  add_compile_definitions(STRATA_AMD_COMMUNITY_DOT_OFF=1)
endif()
# The rows one verify / batch window holds (include/strata/kernels/window_limits.hpp): 8, the 0.2.7 window, or 16 (four pipelined
# groups of 16 slots = 64 slots on a four-card split, docs/BATCHING.md).  Only the CUDA and HIP engines are built from
# these sources with a wide window; the SYCL and Metal builds never read this variable and keep 8.  A window of up to
# 8 rows runs the kernels it always ran, whichever is chosen (the wide instantiations are extra code).  "auto" = 16 for
# the HIP engines (a HIP code object loads on first use, so the extra kernels cost nothing until a wide window forms)
# and 8 for CUDA, which loads every kernel at start (CUDA_MODULE_LOADING=EAGER: the wide kernels would take VRAM from
# the expert cache of every card); -DSTRATA_WINDOW_MAX_T=16 builds the wide CUDA engine.
set(STRATA_WINDOW_MAX_T "auto" CACHE STRING "rows of a verify / batch window: 8, 12, 16 or auto (HIP 16, CUDA 8)")
set_property(CACHE STRATA_WINDOW_MAX_T PROPERTY STRINGS auto 8 12 16)
if(NOT STRATA_WINDOW_MAX_T STREQUAL "auto" AND NOT STRATA_WINDOW_MAX_T STREQUAL "8" AND NOT STRATA_WINDOW_MAX_T STREQUAL "12" AND NOT STRATA_WINDOW_MAX_T STREQUAL "16")
  message(FATAL_ERROR "STRATA_WINDOW_MAX_T must be auto, 8, 12 or 16 (got '${STRATA_WINDOW_MAX_T}')")
endif()
if(STRATA_ENABLE_HIP OR STRATA_ENABLE_CUDA)
  if(STRATA_WINDOW_MAX_T STREQUAL "auto")
    if(STRATA_ENABLE_HIP)
      set(_strata_window_max_t 16)
    else()
      set(_strata_window_max_t 8)
    endif()
  else()
    set(_strata_window_max_t ${STRATA_WINDOW_MAX_T})
  endif()
  add_compile_definitions(STRATA_VERIFY_MAX_T=${_strata_window_max_t})
  message(STATUS "Strata: verify / batch windows hold up to ${_strata_window_max_t} rows")
endif()
# MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K (Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF
# projections, STRATA_DENSE_MMQ).  On by default since MODEL-UPDATE Phase 1: on CUDA they are their own library
# (strata_kq, loaded only for a model with these formats: the engine loads every kernel of its own binary at start,
# CUDA_MODULE_LOADING=EAGER, and these took ~3.8 MB of VRAM - 3 IQ3_XXS expert slots on a P100 - from every model); on
# HIP a code object loads on first use, so they cost other models nothing there (R9700: the same slots).  Used at run
# time when STRATA_MMQ_KQ=1, which the engine sets by itself on NVIDIA cards of compute capability 8.0 and up (include/strata/core/nv_defaults.hpp) and on
# Linux gfx12 (STRATA_MMQ_KQ=0 turns it off; setup also writes it for UD-Q5_K_XL / UD-Q6_K_XL); other cards keep the FP16 path.  This CMake option only
# decides whether the kernels are BUILT: on in every release build.
option(STRATA_MMQ_TQ_OVERLAY "Gyro trellis MMQ prompt kernels: the patched ggml-cuda headers of third_party/strata-ggml/ggml/src/ggml-cuda-tq (with -DSTRATA_GGML_DIR: an overlay of that llama.cpp's headers written at configure time, cmake/strata_ggml_tq_overlay.cmake)" ON)
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files, and on HIP the dense GGUF projections (STRATA_DENSE_MMQ)" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default since MODEL-UPDATE Phase 1: one more instance per grouped kernel in iq_kernels.cu, +19 KB of sm_60 code,
# measured to leave every expert-cache slot count as it was (P100, R9700).  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
# CUDA: the pointer-table MMQ behind STRATA_MMQ_DIRECT=1 (the prompt path's MMQ reads resident experts in their cache
# slots through a Strata-side copy of llama.cpp's MMQ scheduler, src/prefill/moe_mmq_tab.cuh).  Off by default, like
# the K-quant options above: the engine loads every kernel when it starts (CUDA_MODULE_LOADING=EAGER), and these nine
# instances (+15 MB of binary) take 36-38 MiB of VRAM per GPU on sm_86 - expert slots every model would lose, with
# STRATA_MMQ_DIRECT unset too.  They also add ~70% of the upstream MMQ instances' compile time per CUDA arch (175 s
# of CPU for the nine at sm_86 alone).  Off, the build's device code is byte for byte the one without this feature.  CUDA
# only (not HIP, not gfx906); switched off at configure time when the llama.cpp tree's mmq.cuh is not one the copy
# was verified against.
option(STRATA_MMQ_TAB "Build the pointer-table prompt MMQ for STRATA_MMQ_DIRECT=1 (CUDA; ~37 MiB VRAM per GPU)" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# Tests are on by default when the build can run them.  What they need: their sources (src/**/*_test.cpp, tests/core,
# bench/micro; a source drop without them has nothing to build) and the x86 CPU kernels, which most of the CPU suite drives
# (arm64 stays off by default: -DSTRATA_BUILD_TESTS=ON still builds what builds there).  GPU tests are registered only
# when STRATA_ENABLE_CUDA / STRATA_ENABLE_HIP is on, and the configure summary below says which kinds were registered, so
# a missing GPU is a stated skip and never a silent pass.  Release builds say -DSTRATA_BUILD_TESTS=OFF themselves
# (packaging/, setup.py, Dockerfile, spark/).
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/core AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro AND STRATA_CPU_X86)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
# A test that needs a model file (a real expert blob from a pack, which is not in the source tree): registered as the real test
# when the file is there, and otherwise as a stated skip that ctest lists as "Skipped" with this reason, never as a pass of nothing
# and not as a failure on a machine that has no pack.
function(strata_add_pack_test name needs)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${needs}")
    add_test(NAME ${name} COMMAND ${ARGN} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo "SKIPPED: ${name} needs ${needs} (a Q2_0 expert pack, a model file), which is not in this tree")
    set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")
  endif()
endfunction()
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# #2027: defined here, before the GPU section reads it to register the native-expert tests (it was defined at the
# native-experts section far below, so a first configure registered four tests fewer than the second)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_split_failure_test src/core/conversation_split_failure_test.cpp)
  target_include_directories(conversation_split_failure_test PRIVATE include)
  add_test(NAME conversation_split_failure_test COMMAND conversation_split_failure_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(glm_conversations_test src/poly/conversations_test.cpp src/poly/spill.cpp)
  add_test(NAME glm_conversations_test COMMAND glm_conversations_test)
  add_executable(poly_guesses_test src/poly/guesses_test.cpp src/spec/draft_policy.cpp)
  target_include_directories(poly_guesses_test PRIVATE include)
  add_test(NAME poly_guesses_test COMMAND poly_guesses_test)
  add_executable(conversation_file_test src/core/conversation_file_test.cpp src/core/conversation_file.cpp)
  target_include_directories(conversation_file_test PRIVATE include)
  add_test(NAME conversation_file_test COMMAND conversation_file_test)
  add_executable(session_save_reclaim_test src/core/session_save_reclaim_test.cpp src/core/conversation_file.cpp)
  target_include_directories(session_save_reclaim_test PRIVATE include)
  add_test(NAME session_save_reclaim_test COMMAND session_save_reclaim_test)
endif()

if(STRATA_HIP_GFX906)
  strata_hip_gfx906_toolchain()
elseif(STRATA_ENABLE_CUDA)
  strata_cuda_toolchain()
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target ${STRATA_HIP_BLAS_TARGETS})
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  add_executable(batch_telemetry_test src/core/batch_telemetry_test.cpp)
  target_include_directories(batch_telemetry_test PRIVATE include)
  add_test(NAME batch_telemetry_test COMMAND batch_telemetry_test)
  # #5697: an evicted parked conversation gives its RAM back to the system (glibc's heap; RSS before and after)
  add_executable(conversation_heap_test src/core/conversation_heap_test.cpp)
  target_include_directories(conversation_heap_test PRIVATE include)
  add_test(NAME conversation_heap_test COMMAND conversation_heap_test)
  # #5718 follow-up: the stall report's list of GPU streams a host may wait on, and its lines (faked queries, no GPU)
  add_executable(stream_watch_test tests/core/stream_watch_test.cpp)
  target_include_directories(stream_watch_test PRIVATE include)
  add_test(NAME stream_watch_test COMMAND stream_watch_test)
  add_executable(logit_bias_test src/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE include)
  add_test(NAME logit_bias_test COMMAND logit_bias_test)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the model families (qwen4exp, qwen35moe): the guards, the geometry from metadata, the layout tables (synthetic)
  add_executable(model_family_test tests/core/model_family_test.cpp)
  target_link_libraries(model_family_test PRIVATE strata_artifact)
  add_test(NAME model_family_test COMMAND model_family_test)
  # how --batch cuts a prompt read into pieces (a layer split with no slot decoding must still reach a BYIELD)
  add_executable(batch_read_test tests/program/batch_read_test.cpp)
  target_include_directories(batch_read_test PRIVATE include)
  add_test(NAME batch_read_test COMMAND batch_read_test)
  # the model's geometry from its metadata, and the check against the kernels' compiled shape (synthetic)
  add_executable(gguf_geometry_test tests/core/gguf_geometry_test.cpp)
  target_link_libraries(gguf_geometry_test PRIVATE strata_artifact)
  add_test(NAME gguf_geometry_test COMMAND gguf_geometry_test)

  # the hipBLASLt tuning header's architecture gate: RDNA4 siblings share solution IDs, RDNA3 siblings share
  # theirs, cross-family pairs and version mismatches stay rejected (synthetic tables: no model, no GPU)
  add_executable(tuning_header_arch_family_test tests/prefill/tuning_header_arch_family_test.cpp)
  target_include_directories(tuning_header_arch_family_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
  add_test(NAME tuning_header_arch_family_test COMMAND tuning_header_arch_family_test)

  # #5722: each layer-split participant's prompt loan priced on its own device (faked devices, no GPU)
  add_executable(split_part_price_test tests/program/split_part_price_test.cpp)
  target_include_directories(split_part_price_test PRIVATE include)
  add_test(NAME split_part_price_test COMMAND split_part_price_test)
  # STRATA_RESIDENT_PIN read for the automatic RAM copy too, and that copy's clamp line names no flag (header-only, no GPU)
  add_executable(resident_ram_copy_test tests/core/resident_ram_copy_test.cpp)
  target_include_directories(resident_ram_copy_test PRIVATE include)
  add_test(NAME resident_ram_copy_test COMMAND resident_ram_copy_test)
  # the auto layer split's minimum stage size (#1616): header-only rule, no GPU
  add_executable(split_rules_test tests/program/split_rules_test.cpp)
  target_include_directories(split_rules_test PRIVATE include)
  add_test(NAME split_rules_test COMMAND split_rules_test)

  # #3797: the out-of-memory report at start (per card free/total, the largest other programs): the text is a pure
  # function; the driver query (NVML, loaded at run time) is only checked to be safe to call. No GPU needed.
  add_executable(gpu_report_test tests/program/gpu_report_test.cpp)
  target_include_directories(gpu_report_test PRIVATE include)
  target_link_libraries(gpu_report_test PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME gpu_report_test COMMAND gpu_report_test)

  # #3702: the later-stage reserve of the layer split search (header-only arithmetic, pinned against the old expression)
  add_executable(stage_reserve_test tests/program/stage_reserve_test.cpp)
  target_include_directories(stage_reserve_test PRIVATE include)
  add_test(NAME stage_reserve_test COMMAND stage_reserve_test)

  # the layer split's spill walk (header-only arithmetic: who is short, which step comes next)
  add_executable(split_walk_test tests/program/split_walk_test.cpp)
  target_include_directories(split_walk_test PRIVATE include)
  add_test(NAME split_walk_test COMMAND split_walk_test)

  # the prompt planner's host-memory model (#2824): header-only, no GPU
  add_executable(prefill_plan_test tests/program/prefill_plan_test.cpp)
  target_include_directories(prefill_plan_test PRIVATE include)
  add_test(NAME prefill_plan_test COMMAND prefill_plan_test)

  # the default CPU prefill share is not armed when every expert is resident (#1595): header-only rule, no GPU
  add_executable(share_rules_test tests/prefill/share_rules_test.cpp)
  target_include_directories(share_rules_test PRIVATE include)
  add_test(NAME share_rules_test COMMAND share_rules_test)

  # STRATA_SELECT_TWO_PASS: the bound between the single-pass and 3xTF32 QSA block scores and the cut-off rule that
  # keeps the selection (header-only, no GPU)
  add_executable(qsa_two_pass_test tests/prefill/qsa_two_pass_test.cpp)
  target_include_directories(qsa_two_pass_test PRIVATE include)
  add_test(NAME qsa_two_pass_test COMMAND qsa_two_pass_test)
  # STRATA_SELECT_TWO_PASS: the request's parse and the line it prints once where the scorer cannot run (no GPU)
  add_executable(qsa_two_pass_status_test tests/prefill/qsa_two_pass_status_test.cpp)
  target_include_directories(qsa_two_pass_status_test PRIVATE include)
  add_test(NAME qsa_two_pass_status_test COMMAND qsa_two_pass_status_test
           "${CMAKE_CURRENT_BINARY_DIR}/qsa_two_pass_status_test.log")
           
  # STRATA_PREFILL_ROUTED_STREAM: the prompt stream's issuer walk and route gate, on a simulated link and consumer
  # (header-only, no GPU)
  add_executable(routed_stream_test tests/prefill/routed_stream_test.cpp)
  target_include_directories(routed_stream_test PRIVATE include)
  target_link_libraries(routed_stream_test PRIVATE Threads::Threads)
  add_test(NAME routed_stream_test COMMAND routed_stream_test)

  # PR #4006: STRATA_PREFILL_TIMING does not record a mark in the phase already running (header-only, no GPU)
  add_executable(timer_marks_test tests/prefill/timer_marks_test.cpp)
  target_include_directories(timer_marks_test PRIVATE include)
  add_test(NAME timer_marks_test COMMAND timer_marks_test)

  # PR #5108: the MMQ row table's over-read margin follows the backend tile bound; every fixup read stays inside (header-only, no GPU)
  add_executable(mmq_ids_margin_test tests/prefill/mmq_ids_margin_test.cpp)
  target_include_directories(mmq_ids_margin_test PRIVATE include)
  add_test(NAME mmq_ids_margin_test COMMAND mmq_ids_margin_test)

  # PR #4844: the adaptive tier's copy-in range across serial, pipelined and drained rounds (header-only, no GPU)
  add_executable(adaptive_range_test tests/program/adaptive_range_test.cpp)
  target_include_directories(adaptive_range_test PRIVATE include)
  add_test(NAME adaptive_range_test COMMAND adaptive_range_test)

  # P0.1 (lite): STRATA_MANIFEST, the machine-readable record of the selected paths, tensor formats, engine and budgets (header-only, no GPU)
  add_executable(run_manifest_test tests/core/run_manifest_test.cpp)
  target_include_directories(run_manifest_test PRIVATE include)
  add_test(NAME run_manifest_test COMMAND run_manifest_test)

  # the Windows HIP auto prefill chunk cap (#1630) and the commit-limit verdict (#1607): header-only, no GPU
  add_executable(host_limits_test tests/platform/host_limits_test.cpp)
  target_include_directories(host_limits_test PRIVATE include)
  add_test(NAME host_limits_test COMMAND host_limits_test)
  # #4733: an engine started at Low I/O priority goes back to Normal (Windows; else skips)
  add_executable(io_priority_test tests/platform/io_priority_test.cpp src/platform/io_priority.cpp)
  target_include_directories(io_priority_test PRIVATE include)
  add_test(NAME io_priority_test COMMAND io_priority_test)
  set_tests_properties(io_priority_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
strata_gpu_engine()

strata_cpu_kernels()

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from the tree vendored in
# third_party/strata-ggml (frozen at the llama.cpp commit in third_party/ggml/VERSION.txt, see its UPSTREAM.md): no download.
# -DSTRATA_GGML_DIR=<llama.cpp checkout> still builds from that checkout instead (an override kept for two releases).
strata_ggml_resolve("${CMAKE_CURRENT_SOURCE_DIR}")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  strata_ggml_configure_root()
  strata_cpu_native_experts()
  strata_ggml_mmq_types()
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    strata_cuda_mmq()
  elseif(STRATA_PREFILL_MMQ)
    strata_hip_mmq()
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  # the NVFP4 AVX-512 and AVX2 rows vs ggml-cpu's vec_dot, and their speed (CPU only; optional arg: a pack's experts.bin)
  add_executable(nvfp4_avx512_parity src/kernels/cpu/nvfp4_avx512_parity.cpp)
  target_link_libraries(nvfp4_avx512_parity PRIVATE strata_kernels_cpu)
  if(STRATA_ISA_FLOOR)   # ggml-cpu without AVX2 / FMA: its dot is not the AVX2 one, so no bit-equality check (cmake/backends/cpu.cmake)
    target_compile_definitions(nvfp4_avx512_parity PRIVATE STRATA_TEST_GGML_WITHOUT_AVX2=1)
  endif()
  add_test(NAME nvfp4_cpu_parity COMMAND nvfp4_avx512_parity)   # random blocks; the AVX2 rows bit-equal to ggml's
  set_tests_properties(nvfp4_cpu_parity PROPERTIES TIMEOUT 120)
  if(STRATA_ENABLE_CUDA)   # the NVFP4 decode kernels vs ggml's reference
    add_executable(nvfp4_expert_gpu_parity src/kernels/nvfp4_expert_gpu_parity.cpp)
    target_link_libraries(nvfp4_expert_gpu_parity PRIVATE strata_kernels ggml-base CUDA::cudart)
    # the decode experts' weight bandwidth on random blobs of a format pair, and an output hash (bit-for-bit checks)
    add_executable(nvfp4_decode_bench src/kernels/nvfp4_decode_bench.cpp)
    target_link_libraries(nvfp4_decode_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    # strata_engine: the --load-check mode runs the engine's start-up checks (NativeDense, check_experts_gguf)
    target_link_libraries(native_expert_parity PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
      # STRATA_EXPERT_NIBBLE (include/strata/core/expert_nibble.hpp): the 4-bit gate/up expert slots against the
      # production kernels and the GGUF blobs, bit for bit, on a real model's layers (needs a GP100); --header: the
      # contract's host half, no GPU
      add_executable(iq_nibble_parity src/kernels/iq_nibble_parity.cpp)
      target_include_directories(iq_nibble_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(iq_nibble_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ggml-base ${_strata_gpu_runtime_target})
      if(STRATA_BUILD_TESTS)
        add_test(NAME iq_nibble_parity_header COMMAND iq_nibble_parity --header)
        # the synthetic experts of every gate/up x down format pair, no model (skips with 77 on a card that is not a GP100)
        add_test(NAME iq_nibble_parity_synthetic COMMAND iq_nibble_parity --synthetic-only)
        set_tests_properties(iq_nibble_parity_synthetic PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
      endif()
    endif()
    # MODEL-UPDATE: the dense Q8_0 GEMVs at the engine's shapes against the memory bandwidth (a benchmark, no test)
    add_executable(q8_gemv_bench src/kernels/q8_gemv_bench.cpp)
    target_include_directories(q8_gemv_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q8_gemv_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    # EXPDEC (docs/EXPERTDEC.md): the grouped experts' default kernels against STRATA_EXPDEC=1 at T = 1..4, bit for bit
    # and in microseconds, on synthetic blobs read cold from DRAM
    add_executable(expdec_bench EXCLUDE_FROM_ALL src/kernels/expdec_bench.cpp)
    target_include_directories(expdec_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(expdec_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
      # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
      # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
      # q4_K/q5_0 and q5_0/q8_0: a plain llama-quantize "_M" tier GGUF (not Unsloth's dynamic method) commonly
      # puts Q5_0 on down-projections (sometimes gate/up too) where this model's native_experts.txt had no GPU
      # kernel for it before; q5_0/q5_0 (both roles Q5_0) is left out of this list since stacking the same
      # lower-precision format on both roles pushes the GPU path over this test's 3% threshold (quantization
      # noise, not a kernel bug -- see the PR).
      # q4_0/q4_0 and q4_0/q4_1: a plain llama-quantize Q4_0 file (bartowski's Qwen3.8-Flash-Next-Q4_0: gate/up Q4_0,
      # down Q4_0 on 42 layers and Q4_1 on 6)
      # q6_K/q8_0: Unsloth UD-Q6_K_XL gate/up Q6_K with Q8_0 down (docs/UNSLOTH_Q6.md). Layer 2 is q8_0/q8_0, already listed.
      set(_strata_parity_pairs q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0 q4_0/q4_0 q4_0/q4_1)
      if(STRATA_Q6K_EXPERTS)
        list(APPEND _strata_parity_pairs q6_K/q8_0)
      else()   # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
        add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
        set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
      endif()
      foreach(_pair ${_strata_parity_pairs})
        string(REPLACE "/" "_" _name "${_pair}")
        add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
        set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
      endforeach()
      add_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
      # Unsloth v3.1 Qwen3.8-Flash-Next sets (UD-IQ1_M .. UD-IQ4_XS, round 0214): every gate/up x down pair their layers use,
      # at the real shapes (hidden 2560, FF 640): gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ4_XS (+ IQ1_M, the GSQ-RCO packs'),
      # down Q2_0 IQ4_NL Q5_0 Q8_0.  Q5_0 down (UD-IQ4_XS 33 layers, UD-Q3_K_XL 20) had no test on this model before.
      foreach(_gu iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq4_xs)
        foreach(_dn q2_0 iq4_nl q5_0 q8_0)
          add_test(NAME native_expert_parity_v31_${_gu}_${_dn} COMMAND native_expert_parity --synthetic ${_gu}/${_dn})
          set_tests_properties(native_expert_parity_v31_${_gu}_${_dn} PROPERTIES TIMEOUT 180)
        endforeach()
      endforeach()
      # the dense projections (Q5_0, Q6_K, Q5_K, Q4_K, Q8_0, IQ4_XS, IQ4_NL ...): decode MMVQ and the prompt's dequantizers
      add_executable(native_dense_parity src/kernels/native_dense_parity.cpp)
      target_include_directories(native_dense_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(native_dense_parity PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME native_dense_parity COMMAND native_dense_parity)
      set_tests_properties(native_dense_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # APR: the trellis expert types and the Hadamard activation rotation, when ggml comes from the agention fork
      # (the only tree with these types; marked by its ggml-cuda/tq.cuh)
      if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/tq.cuh")
        set(_strata_tq_pairs tqk6/tqk7 tq2_t/tq2_t)
        if(EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqp5.cu")   # Gyro-L's pair trellis
          list(APPEND _strata_tq_pairs tqp5/tqp5 tqp6/tqp6 tqp7/tqp7)
        endif()
        foreach(_pair ${_strata_tq_pairs})
          string(REPLACE "/" "_" _name "${_pair}")
          add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
          add_test(NAME native_expert_parity_hadamard_${_name} COMMAND native_expert_parity --synthetic-hadamard ${_pair})
          set_tests_properties(native_expert_parity_${_name} native_expert_parity_hadamard_${_name} PROPERTIES TIMEOUT 300)
        endforeach()
      endif()
      # #290: the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit
      add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    endif()
  endif()
  # Round 0214 (the new Unsloth Qwen3.8-Flash-Next quants UD-IQ1_M .. UD-IQ4_XS), CUDA and HIP: every expert type pair the
  # sets use at 2560/640 (GPU decode kernels, the CPU pool and the prompt path's dequantizers against ggml), and the dense
  # tensor types (Q5_0, Q6_K, the K-quants, IQ4_NL, IQ4_XS, Q2_0) through native_mmvq.
  if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    set(_strata_new_gu iq1_m iq2_xxs iq2_xs iq2_s q2_0 iq3_xxs iq4_xs)
    set(_strata_new_dn q2_0 iq4_nl q5_0 q8_0)
    foreach(_g ${_strata_new_gu})
      foreach(_d ${_strata_new_dn})
        add_test(NAME native_expert_parity_new_${_g}_${_d} COMMAND native_expert_parity --synthetic ${_g}/${_d})
        set_tests_properties(native_expert_parity_new_${_g}_${_d} PROPERTIES TIMEOUT 300)
      endforeach()
    endforeach()
    add_executable(native_dense_types_parity src/kernels/native_dense_types_parity.cpp)
    target_include_directories(native_dense_types_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_dense_types_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
    add_test(NAME native_dense_types_parity COMMAND native_dense_types_parity)
    set_tests_properties(native_dense_types_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()
  # #296's real-weight checks (OrcaRouter Q4_K_S files; docs/ORCA_Q4_K_S.md) - test programs, not part of a release
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
    add_executable(ple_q5_parity src/kernels/ple_q5_parity.cpp)
    target_link_libraries(ple_q5_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    add_executable(q5_projection_parity src/kernels/q5_projection_parity.cpp)
    target_include_directories(q5_projection_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q5_projection_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    # #291: the FP8 PLE table (tools/ple_fp8_pack.py) through PleTable vs the checkpoint's own values
    add_executable(ple_fp8_parity src/kernels/ple_fp8_parity.cpp)
    target_link_libraries(ple_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
endif()

strata_cpu_tests()

# CUDA release kernels must not spill or keep a per-thread array in local memory: every access there is a round trip
# to device memory in the hottest loops the engine has, and nothing on the build machine says so without -Xptxas -v
# or a GPU.  tools/check_cuda_spills.py reads the resource usage the compiler wrote into the kernel libraries
# (cuobjdump: STACK and LOCAL per entry function and per sm_) and fails on any kernel outside its ALLOWED table, which
# carries the measured allowance of the kernels that hold such an array on purpose.  CUDA only: HIP has no cuobjdump.
strata_cuda_spill_guard()

# The opt-in lifecycle is a host-only test, including on non-CUDA builds.
if(STRATA_BUILD_TESTS)
  add_executable(poly_hip_value_split_policy_test src/poly/hip_value_split_policy_test.cpp)
  add_test(NAME poly_hip_value_split_policy_test COMMAND poly_hip_value_split_policy_test)
  add_executable(poly_gdn_mma_control_test src/poly/gdn_mma_control_test.cpp)
  add_test(NAME poly_gdn_mma_control_test COMMAND poly_gdn_mma_control_test)
  add_executable(poly_stream_stage_test src/poly/stage_copy_test.cpp)
  add_test(NAME poly_stream_stage_test COMMAND poly_stream_stage_test)
endif()

# Shared prepared-input GDN recurrence. Either engine may link this library;
# only the Poly Qwen35 adapter is integrated initially. Keep FP32/BF16 hi+lo math
# separate from the MMQ targets' -use_fast_math compilation options.
if(STRATA_NATIVE_EXPERTS AND STRATA_ENABLE_CUDA)
  if(STRATA_GGML_DIR)
    set(_strata_gdn_ggml ${STRATA_GGML_DIR}/ggml)
  else()
    set(_strata_gdn_ggml ${_strata_ggml_root}/ggml)
  endif()
  add_library(strata_gdn_mma STATIC src/kernels/cuda/gdn_mma.cu)
  target_include_directories(strata_gdn_mma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${_strata_gdn_ggml}/include ${_strata_gdn_ggml}/src ${_strata_gdn_ggml}/src/ggml-cuda)
  target_link_libraries(strata_gdn_mma PRIVATE ggml-base ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS)
    add_executable(poly_gdn_mma_policy_test src/kernels/gdn_mma_policy_test.cpp)
    target_include_directories(poly_gdn_mma_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME poly_gdn_mma_policy_test COMMAND poly_gdn_mma_policy_test)
    add_executable(gdn_mma_test src/kernels/gdn_mma_test.cu)
    target_link_libraries(gdn_mma_test PRIVATE strata_gdn_mma ${_strata_gpu_runtime_target})
    add_test(NAME gdn_mma_test COMMAND gdn_mma_test)
    set_tests_properties(gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

# ---- strata-poly: GLM 5.3 Flash (llama.cpp's glm5-next), read from its GGUF ----
#
# A second engine beside `strata`.  It shares the quantized matvec kernels and the GGUF reader, and takes the
# CPU experts' dot products from ggml-cpu.  On HIP it is the same CUDA-shaped code through the hip_compat headers
# (the wave32 RDNA backend; gfx906's wave64 compat layer is not supported here).
if(STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND NOT STRATA_HIP_GFX906)
  # The prompt path's quantized products: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the types the model's
  # GGUFs keep their matrices in (the list in src/poly/mmq.cu), from the checkout strata_mmq is compiled from and
  # with the same host symbols.  A library of its own, so that what `strata` multiplies with MMQ stays as it is.
  if(NOT _strata_mmq)   # a HIP build without STRATA_PREFILL_MMQ has not set these up yet
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${_strata_ggml_root}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
  endif()
  set(_strata_poly_mmq_instances "")
  foreach(_t q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl
             iq4_xs)
    list(APPEND _strata_poly_mmq_instances ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
  endforeach()
  # MXFP4: on RTX 50 llama.cpp's tiles use the FP4 MMA with block scales, which exists only in the arch-specific
  # target, so on CUDA its instance alone is built with 12x -> 12xa, as Strata's W4A4 unit is.  A Blackwell other
  # than 12.0 has no image of it, and Mmq::fits() leaves MXFP4 out there.
  set(_strata_poly_mxfp4 ${_strata_mmq}/template-instances/mmq-instance-mxfp4.cu)
  if(STRATA_ENABLE_HIP)
    list(APPEND _strata_poly_mmq_instances ${_strata_poly_mxfp4})
  else()
    set(_strata_poly_fp4_archs ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_poly_mmq_mxfp4 OBJECT ${_strata_poly_mxfp4})
    set_target_properties(strata_poly_mmq_mxfp4 PROPERTIES CUDA_ARCHITECTURES "${_strata_poly_fp4_archs}")
    target_include_directories(strata_poly_mmq_mxfp4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    list(APPEND _strata_poly_mmq_instances $<TARGET_OBJECTS:strata_poly_mmq_mxfp4>)
  endif()
  if(STRATA_ENABLE_HIP)
    # CMake does not infer HIP from the .cu suffixes outside src/ (and drops a .cu source it has no language
    # for), and ggml's MMQ sources take its native HIP path (vendors/hip.h) instead of the hip_compat headers,
    # as strata_mmq's does.
    set_source_files_properties(${_strata_mmq}/quantize.cu ${_strata_poly_mmq_instances} PROPERTIES LANGUAGE HIP)
  endif()
  add_library(strata_poly_mmq STATIC src/poly/mmq.cu src/poly/fattn.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu
              ${_strata_poly_mmq_instances})
  target_include_directories(strata_poly_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
  if(MSVC)
    target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  endif()
  if(STRATA_ENABLE_HIP)
    target_compile_definitions(strata_poly_mmq PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base hip::host)
  else()
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
  endif()

  if(STRATA_ENABLE_CUDA)
    target_compile_definitions(strata_poly_mmq PUBLIC STRATA_POLY_GDN_MMA=1)
    target_link_libraries(strata_poly_mmq PUBLIC strata_gdn_mma)

    add_executable(poly_gdn_mma_test src/poly/gdn_mma_test.cu src/poly/kernels.cu)
    target_include_directories(poly_gdn_mma_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(poly_gdn_mma_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
    add_test(NAME poly_gdn_mma_test COMMAND poly_gdn_mma_test)
    set_tests_properties(poly_gdn_mma_test PROPERTIES SKIP_RETURN_CODE 77)
    # Optional model/state check; run by hand with an MTP GGUF and frozen prompt IDs.
    add_executable(poly_gdn_mma_state_test src/poly/gdn_mma_state_test.cpp src/poly/engine.cpp src/poly/glm5next.cpp
                   src/poly/qwen35.cpp src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp
                   src/poly/flint.cpp src/poly/kolibri1.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu
                   src/poly/kernels_dsv4.cu src/core/conversation_memory.cpp)
    target_include_directories(poly_gdn_mma_state_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(poly_gdn_mma_state_test PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                         ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target} CUDA::cuda_driver Threads::Threads)
  endif()

  add_executable(strata-poly src/poly/main.cpp src/poly/engine.cpp src/poly/glm5next.cpp src/poly/qwen35.cpp
                            src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp src/poly/flint.cpp
                            src/poly/kolibri1.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu src/poly/kernels_dsv4.cu
                            src/poly/spill.cpp src/core/conversation_memory.cpp)
  target_include_directories(strata-poly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  # the driver's virtual memory calls: the attention caches grow with the conversation; on HIP they are HIP's
  # memory management calls, which the HIP runtime itself provides (hip_compat/cuda.h)
  if(STRATA_ENABLE_HIP)
    set(_strata_poly_driver "")
  else()
    set(_strata_poly_driver CUDA::cuda_driver)
  endif()
  target_link_libraries(strata-poly PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                           ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                           ${_strata_poly_driver}
                                           Threads::Threads)
  # needs the model file, so it is run by hand: poly_expert_parity MODEL.gguf
  add_executable(poly_expert_parity src/poly/expert_parity.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu)
  target_include_directories(poly_expert_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
  target_link_libraries(poly_expert_parity PRIVATE strata_poly_mmq strata_kernels ggml-cpu ggml-base
                                                  ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
  # a read with the draft block repeats bit for bit; needs the model file, so it is run by hand:
  # poly_draft_race_test MODEL.gguf [RUNS] [LAYER_SPLIT]
  add_executable(poly_draft_race_test src/poly/draft_race_test.cpp src/poly/engine.cpp src/poly/glm5next.cpp
                                      src/poly/qwen35.cpp src/poly/gemma4.cpp src/poly/deepseek4.cpp
                                      src/poly/lfm2moe.cpp src/poly/flint.cpp src/poly/kolibri1.cpp
                                      src/poly/cpu_experts.cpp src/poly/kernels.cu src/poly/kernels_dsv4.cu
                                      src/core/conversation_memory.cpp)
  target_include_directories(poly_draft_race_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_draft_race_test PRIVATE strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                                    ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    ${_strata_poly_driver} Threads::Threads)
  # a prompt batch's routing on the card against the hosts' routing; needs a card, so it is run by hand
  add_executable(poly_route_test src/poly/route_test.cu src/poly/kernels.cu)
  target_include_directories(poly_route_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_route_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # the fused residual sum, RMS norm and Q8_1 quantization against the three launches; needs a card, run by hand
  add_executable(poly_norm_q8_test src/poly/norm_q8_test.cu src/poly/kernels.cu)
  target_include_directories(poly_norm_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories})
  target_link_libraries(poly_norm_q8_test PRIVATE strata_kernels ${_strata_gpu_blas_target}
                                                  ${_strata_gpu_runtime_target})
  # a prompt batch's products with few outputs against the same sums on the host; needs a card, run by hand
  add_executable(poly_gemm_small_test src/poly/gemm_small_test.cu src/poly/kernels.cu)
  target_include_directories(poly_gemm_small_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_gemm_small_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # attend()'s blocks of rows against its other kernels and itself; needs a card, so it is run by hand
  add_executable(poly_attend_test src/poly/attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                     ${_strata_gpu_include_directories})
  target_link_libraries(poly_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_executable(poly_hip_value_split_test src/poly/hip_value_split_test.cu src/poly/kernels.cu)
  target_include_directories(poly_hip_value_split_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${_strata_gpu_include_directories})
  target_link_libraries(poly_hip_value_split_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  add_test(NAME poly_hip_value_split_test COMMAND poly_hip_value_split_test)
  set_tests_properties(poly_hip_value_split_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
  # attend_batch() against attend() with Flint's heads, sinks and window ring; needs a card
  add_executable(poly_sink_attend_test src/poly/sink_attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_sink_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                          ${_strata_gpu_include_directories})
  target_link_libraries(poly_sink_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # fattn_q8() and fattn_f16() (llama.cpp's flash-attention kernel) against attend_batch(), and with `dec` fdec_q8()
  # against attend(); needs a card, run by hand
  add_executable(poly_fattn_test src/poly/fattn_test.cu src/poly/kernels.cu)
  target_include_directories(poly_fattn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_fattn_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # LFM2's short convolution against the same arithmetic on the host; needs a card, so it is run by hand
  add_executable(poly_shortconv_test src/poly/shortconv_test.cu src/poly/kernels.cu)
  target_include_directories(poly_shortconv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                        ${_strata_gpu_include_directories})
  target_link_libraries(poly_shortconv_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # DeepSeek V4's kernels against the same arithmetic on the host; needs a card, so it is run by hand
  add_executable(poly_dsv4_test src/poly/dsv4_test.cu src/poly/kernels_dsv4.cu)
  target_include_directories(poly_dsv4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                   ${_strata_gpu_include_directories})
  target_link_libraries(poly_dsv4_test PRIVATE ${_strata_gpu_runtime_target})
  # the pool of the CPU experts, without a model
  add_executable(poly_workers_test src/poly/workers_test.cpp src/poly/cpu_experts.cpp)
  target_include_directories(poly_workers_test PRIVATE ${_strata_gg}/include)
  target_link_libraries(poly_workers_test PRIVATE ggml-cpu ggml-base Threads::Threads)
  add_test(NAME poly_workers_test COMMAND poly_workers_test)
  # the card's MXFP4 product against ggml's dequantizer on random blocks, without a model; needs a card, run by hand
  add_executable(poly_mxfp4_test src/poly/mxfp4_test.cu)
  target_include_directories(poly_mxfp4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gg}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_mxfp4_test PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
  # the Q8_0 KV cache kernels (store_q8 / store_q8_at) against a host reference (GPU, synthetic, no model)
  if(STRATA_BUILD_TESTS)
    add_executable(poly_kv_q8_test src/poly/kv_q8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
    target_link_libraries(poly_kv_q8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
    add_test(NAME poly_kv_q8_test COMMAND poly_kv_q8_test)
  endif()
  # --kv int8/int8k's kernels (store_int8 / attend_int8) against ggml's own quantizer and against attend() over the
  # same numbers in fp16.  The reference is ggml-cpu's quantize_row_q8_0_ref, hence the ggml links; under HIP the
  # mma tier's probes are behind STRATA_USE_HIP (the dp4a tier is what runs there anyway).
  if(STRATA_BUILD_TESTS)
    add_executable(poly_kv_int8_test src/poly/kv_int8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_int8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories} ${_strata_gg}/include)
    target_link_libraries(poly_kv_int8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  ggml-cpu ggml-base Threads::Threads)
    add_test(NAME poly_kv_int8_test COMMAND poly_kv_int8_test)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---- HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) is compiled as HIP ----
strata_hip_gfx906_sources()

# ---- DeepSeek-V4.1 on a box with no GPU toolkit ------------------------------------------------
# The whole DSV41 block above lives inside if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP), so with neither backend on
# -DSTRATA_BUILD_DSV41=ON was silently unused and the host-side benchmark could not be built at all (found on p1004,
# 2026-10-23: the CPU lane is exactly what a box without a usable GPU needs to measure).  This branch builds the one target
# that has no device code, so the CPU-lane A/B runs anywhere.
strata_cpu_dsv41_host_bench()

# ---- strata-split: the opt-in engine that splits every layer across 2-4 NVIDIA cards (src/split, from sFNX, MIT, John Fink) ----
# OFF by default, Linux + CUDA or HIP only, no other target depends on it; see src/split/CMakeLists.txt and docs/SPLIT_ENGINE.md.
option(STRATA_BUILD_SPLIT "Build strata-split: sFNX's split-every-layer engine, host-memory exchange (Linux, CUDA or HIP gfx1201, 2-4 cards)" OFF)
if(STRATA_BUILD_SPLIT)
  add_subdirectory(src/split)
endif()

# What this configure registered, so a missing capability is a stated skip (tools/check prints this line too).
if(STRATA_BUILD_TESTS)
  set(_strata_test_kinds "cpu")
  if(STRATA_ENABLE_CUDA)
    string(APPEND _strata_test_kinds ", cuda")
  elseif(STRATA_ENABLE_HIP)
    string(APPEND _strata_test_kinds ", hip")
  else()
    string(APPEND _strata_test_kinds " (gpu tests not registered: neither STRATA_ENABLE_CUDA nor STRATA_ENABLE_HIP is on)")
  endif()
  if(NOT STRATA_NATIVE_EXPERTS)
    string(APPEND _strata_test_kinds " (native-expert tests not registered: STRATA_NATIVE_EXPERTS=OFF)")
  endif()
  message(STATUS "strata tests: ON - ${_strata_test_kinds}")
else()
  message(STATUS "strata tests: OFF (STRATA_BUILD_TESTS=OFF)")
endif()
