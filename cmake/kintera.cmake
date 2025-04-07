include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME kintera)
set(REPO_URL "https://github.com/chengcli/kintera-dev")
set(REPO_TAG "dev")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${kintera_SOURCE_DIR})
