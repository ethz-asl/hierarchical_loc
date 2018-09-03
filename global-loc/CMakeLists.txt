cmake_minimum_required(VERSION 2.8)
project(global_loc)

find_package(catkin_simple REQUIRED)
catkin_simple()

set(SOURCES
  src/place-retrieval.cc
)

# Protobufs
set(PROTO_DEFNS proto/global-loc/descriptor_index.proto)
PROTOBUF_CATKIN_GENERATE_CPP2("proto" PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
cs_add_library(${PROJECT_NAME} ${SOURCES} ${PROTO_SRCS} ${PROTO_HDRS})

# Preprocessor macros
add_definitions(-DDATA_ROOT_PATH="${CMAKE_SOURCE_DIR}/data/")
add_definitions(-DMODEL_ROOT_PATH="${CMAKE_SOURCE_DIR}/models/")
add_definitions(-DMAP_ROOT_PATH="${CMAKE_SOURCE_DIR}/maps/")

# Executables
cs_add_executable(build_index app/build-index.cc)
target_link_libraries(build_index ${PROJECT_NAME})
cs_add_executable(time_query app/time-query.cc)
target_link_libraries(time_query ${PROJECT_NAME})

# Test files
cs_add_executable(test_opencv test/test_opencv.cc)
cs_add_executable(test_tensorflow test/test_tensorflow.cc)
cs_add_executable(test_inference test/test_inference.cc)
cs_add_executable(test_build_index test/test_build_index.cc)
target_link_libraries(test_build_index ${PROJECT_NAME})
cs_add_executable(test_query_index test/test_query_index.cc)
target_link_libraries(test_query_index ${PROJECT_NAME})

cs_install()
cs_export()
