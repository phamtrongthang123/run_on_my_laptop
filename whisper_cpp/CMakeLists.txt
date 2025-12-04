cmake_minimum_required(VERSION 3.18)
project(whisper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(USE_EXECUTORCH "Use ExecuTorch instead of LibTorch" OFF)
option(USE_CUDA "Enable CUDA support" OFF)
option(BUILD_STATIC "Build static executable" OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find dependencies
if(USE_EXECUTORCH)
    # ExecuTorch
    find_package(executorch REQUIRED)
    add_definitions(-DUSE_EXECUTORCH)
    set(INFERENCE_LIBS executorch)
else()
    # LibTorch
    # Set Torch_DIR if not found automatically
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

if(NOT USE_EXECUTORCH)
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
message(STATUS "  Use CUDA:       ${USE_CUDA}")
message(STATUS "  Static build:   ${BUILD_STATIC}")
if(NOT USE_EXECUTORCH)
    message(STATUS "  Torch version:  ${Torch_VERSION}")
    message(STATUS "  Torch libraries: ${TORCH_LIBRARIES}")
endif()
message(STATUS "")
