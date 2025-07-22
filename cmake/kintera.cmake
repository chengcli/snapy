include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME kintera)
set(REPO_URL "https://github.com/chengcli/kintera")
set(REPO_TAG "v0.8.1")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${kintera_SOURCE_DIR})

set(KINTERA_LIBRARY kintera::kintera CACHE STRING "Kintera library name")
set(KINTERA_CUDA_LIBRARY kintera::kintera_cu CACHE STRING "Kintera CUDA library name")
set(VAPORS_LIBRARY kintera::vapors CACHE STRING "Kintera Vapors library name")
