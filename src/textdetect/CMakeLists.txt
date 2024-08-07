add_library(textdetect STATIC detect.cpp)

find_package(Torch REQUIRED)
target_link_libraries(textdetect PRIVATE ${TORCH_LIBRARIES})
