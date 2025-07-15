include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(patch_command
    git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/01.eigen.patch)

set(PACKAGE_NAME eigen)
set(REPO_URL "https://gitlab.com/libeigen/eigen.git")
set(REPO_TAG "3.4.0")

#set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "${patch_command}" ON)
include_directories(${eigen_SOURCE_DIR})
