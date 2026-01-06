cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(GPU_BACKEND "HIP" CACHE STRING "GPU backend to use: HIP")

project(gpu_lob LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Building with GPU backend: ${GPU_BACKEND}")

if(WIN32)
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH})
        elseif(DEFINED ENV{HIP_PATH})
            set(ROCM_PATH $ENV{HIP_PATH})
        else()
            set(ROCM_PATH "C:/Program Files/AMD/ROCm/5.7")
        endif()
    endif()

    message(STATUS "ROCm Path: ${ROCM_PATH}")

    set(HIPCC_EXECUTABLE "${ROCM_PATH}/bin/hipcc.exe")
    if(NOT EXISTS ${HIPCC_EXECUTABLE})
        message(FATAL_ERROR "hipcc.exe not found at ${HIPCC_EXECUTABLE}")
    endif()
    message(STATUS "HIP Compiler: ${HIPCC_EXECUTABLE}")

    list(APPEND CMAKE_PREFIX_PATH
        "${ROCM_PATH}"
        "${ROCM_PATH}/hip"
        "${ROCM_PATH}/lib/cmake"
        "${ROCM_PATH}/lib/cmake/hip"
    )

    find_package(hip REQUIRED CONFIG)

    if(NOT DEFINED GPU_TARGETS)
        if(DEFINED ENV{GPU_TARGETS})
            set(GPU_TARGETS $ENV{GPU_TARGETS})
        else()
            set(GPU_TARGETS "gfx1100")
        endif()
        message(STATUS "Using default GPU target: ${GPU_TARGETS}")
    else()
        message(STATUS "Using provided GPU target: ${GPU_TARGETS}")
    endif()

    set(HIP_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/engine.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lob_kernel.cu
    )

    set(HIP_OBJECT_FILES)
    foreach(HIP_SRC ${HIP_SOURCE_FILES})
        get_filename_component(HIP_SRC_NAME ${HIP_SRC} NAME_WE)
        set(HIP_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hip_objs/${HIP_SRC_NAME}.obj)

        add_custom_command(
            OUTPUT ${HIP_OBJ}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/hip_objs
            COMMAND ${HIPCC_EXECUTABLE}
                -DUSE_HIP
                -D__HIP_PLATFORM_AMD__
                --offload-arch=${GPU_TARGETS}
                -std=c++17
                -Xclang --dependent-lib=msvcrt
                -D_DLL
                -I"${ROCM_PATH}/include"
                -I"${ROCM_PATH}/hip/include"
                -c ${HIP_SRC}
                -o ${HIP_OBJ}
            DEPENDS ${HIP_SRC}
            COMMENT "Building HIP object ${HIP_SRC_NAME}.obj"
            VERBATIM
        )

        list(APPEND HIP_OBJECT_FILES ${HIP_OBJ})
    endforeach()

    add_custom_target(hip_kernels ALL DEPENDS ${HIP_OBJECT_FILES})

    add_library(gpu_kernels STATIC ${HIP_OBJECT_FILES})
    set_target_properties(gpu_kernels PROPERTIES
        LINKER_LANGUAGE CXX
    )
    add_dependencies(gpu_kernels hip_kernels)
    target_link_libraries(gpu_kernels PRIVATE hip::host hip::device)

else()
    message(FATAL_ERROR "This CMakeLists.txt is for Windows only. Use CMakeLists.txt.hip for Linux.")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core MODULE
    src/binding.cpp
    WITH_SOABI
)

target_link_libraries(_core PRIVATE gpu_kernels)

target_link_libraries(_core PRIVATE pybind11::headers)

target_link_libraries(_core PRIVATE "${ROCM_PATH}/lib/amdhip64.lib")

set_target_properties(_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_compile_definitions(_core PRIVATE
    USE_HIP
    __HIP_PLATFORM_AMD__
)

install(TARGETS _core
    COMPONENT python_modules
    LIBRARY DESTINATION gpu_lob
    RUNTIME DESTINATION gpu_lob
)

install(DIRECTORY gpu_lob/
    COMPONENT python_modules
    DESTINATION gpu_lob
    FILES_MATCHING PATTERN "*.py"
)

message(STATUS "===========================================")
message(STATUS "GPU LOB Build Configuration (Windows):")
message(STATUS "  GPU Backend:        ${GPU_BACKEND}")
message(STATUS "  GPU Targets:        ${GPU_TARGETS}")
message(STATUS "  ROCm Path:          ${ROCM_PATH}")
message(STATUS "  HIP Compiler:       ${HIPCC_EXECUTABLE}")
message(STATUS "  Python Version:     ${Python_VERSION}")
message(STATUS "  Compiler:           ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Install Prefix:     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===========================================")
