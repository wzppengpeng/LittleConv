include_directories(../../3rdparty/wzp_cpp_lib/ ../../include)
aux_source_directory(. SRC_LIST)
add_executable(cifar10_quick ${SRC_LIST})
target_link_libraries(cifar10_quick licon)