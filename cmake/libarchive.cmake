include(FetchContent)

# ---- libarchive options (tune as needed) ----
set(ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(ENABLE_TEST OFF CACHE BOOL "" FORCE)

set(ENABLE_OPENSSL OFF CACHE BOOL "" FORCE)
set(ENABLE_LZMA OFF CACHE BOOL "" FORCE)
set(ENABLE_ZSTD OFF CACHE BOOL "" FORCE)
set(ENABLE_BZip2 OFF CACHE BOOL "" FORCE)

set(PACKAGE_NAME libarchive)
set(REPO_URL "https://github.com/libarchive/libarchive")
set(REPO_TAG "v3.8.5")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
set(LIBARCHIVE_INCLUDE_DIR
   "${${PACKAGE_NAME}_SOURCE_DIR}/libarchive"
   CACHE PATH "libarchive include directory")
