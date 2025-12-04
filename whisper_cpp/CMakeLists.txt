cmake_minimum_required(VERSION 3.18)
project(whisper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(USE_EXECUTORCH "Use ExecuTorch instead of LibTorch" OFF)
option(USE_CUDA "Enable CUDA support (LibTorch only)" OFF)
option(BUILD_STATIC "Build static executable" OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find dependencies
if(USE_EXECUTORCH)
    message(STATUS "Building with ExecuTorch backend")
    
    # Find ExecuTorch - can be found via:
    # 1. CMAKE_PREFIX_PATH pointing to pip-installed executorch
    # 2. EXECUTORCH_ROOT environment or cmake variable
    
    # Try to find from pip installation first
    if(NOT DEFINED EXECUTORCH_ROOT)
        execute_process(
            COMMAND python3 -c "import executorch; print(executorch.__path__[0])"
            OUTPUT_VARIABLE EXECUTORCH_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    
    if(EXECUTORCH_ROOT)
        message(STATUS "Found ExecuTorch at: ${EXECUTORCH_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_ROOT}/share/cmake")
    endif()
    
    # Find ExecuTorch package
    find_package(executorch REQUIRED)
    
    add_definitions(-DUSE_EXECUTORCH)
    set(INFERENCE_LIBS ${EXECUTORCH_LIBRARIES})
    
else()
    message(STATUS "Building with LibTorch backend")
    
    # LibTorch
    # Set Torch_DIR or CMAKE_PREFIX_PATH to your libtorch installation
    # cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(INFERENCE_LIBS ${TORCH_LIBRARIES})
endif()

# Source files
set(WHISPER_SOURCES
    src/whisper.cpp
    src/audio.cpp
    src/tokenizer.cpp
)

# Header files
set(WHISPER_HEADERS
    include/audio.h
    include/tokenizer.h
)

# Create executable
add_executable(whisper ${WHISPER_SOURCES} ${WHISPER_HEADERS})

# Include directories
target_include_directories(whisper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(USE_EXECUTORCH)
    # ExecuTorch include directories (from find_package)
    target_include_directories(whisper PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
else()
    target_include_directories(whisper PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(whisper PRIVATE ${INFERENCE_LIBS})

# Platform-specific settings
if(WIN32)
    # Windows
    target_compile_definitions(whisper PRIVATE _USE_MATH_DEFINES)
    if(NOT USE_EXECUTORCH)
        # Copy DLLs to output directory
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET whisper POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:whisper>
        )
    endif()
elseif(APPLE)
    # macOS
    set(CMAKE_MACOSX_RPATH ON)
    if(USE_EXECUTORCH)
        # Link against CoreML and Metal for Apple silicon acceleration
        find_library(COREML CoreML)
        find_library(METAL Metal)
        find_library(FOUNDATION Foundation)
        if(COREML AND METAL)
            target_link_libraries(whisper PRIVATE ${COREML} ${METAL} ${FOUNDATION})
        endif()
    endif()
elseif(UNIX)
    # Linux
    target_link_libraries(whisper PRIVATE pthread dl)
endif()

# Static linking option
if(BUILD_STATIC)
    if(NOT APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    endif()
endif()

# Installation
install(TARGETS whisper RUNTIME DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_whisper_to_pte.py
        DESTINATION share/whisper)

# Print configuration summary
message(STATUS "")
message(STATUS "Whisper Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Use ExecuTorch: ${USE_EXECUTORCH}")
if(USE_EXECUTORCH)
    message(STATUS "  ExecuTorch root:     ${EXECUTORCH_ROOT}")
    message(STATUS "  ExecuTorch includes: ${EXECUTORCH_INCLUDE_DIRS}")
    message(STATUS "  ExecuTorch libs:     ${EXECUTORCH_LIBRARIES}")
else()
    message(STATUS "  Use CUDA:       ${USE_CUDA}")
    message(STATUS "  Torch version:  ${Torch_VERSION}")
    message(STATUS "  Torch libraries: ${TORCH_LIBRARIES}")
endif()
message(STATUS "  Static build:   ${BUILD_STATIC}")
message(STATUS "")
