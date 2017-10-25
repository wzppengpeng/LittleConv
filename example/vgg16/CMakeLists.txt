include_directories(../../3rdparty/wzp_cpp_lib/ ../../include)
aux_source_directory(. SRC_LIST)
add_executable(vgg16 ${SRC_LIST})
target_link_libraries(vgg16 licon)