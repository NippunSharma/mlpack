# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
set(SOURCES
  gaussian_process_regression.hpp,
  gaussian_process_regression.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(gaussian_process_regression)
add_python_binding(gaussian_process_regression)
add_julia_binding(gaussian_process_regression)
add_go_binding(gaussian_process_regression)
add_r_binding(gaussian_process_regression)
add_markdown_docs(gaussian_process_regression "cli;python;julia;go;r" "regression")