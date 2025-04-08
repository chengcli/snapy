include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME kintera)
set(REPO_URL "https://github.com/cshsgy/kintera")
set(REPO_TAG "main")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${kintera_SOURCE_DIR})
