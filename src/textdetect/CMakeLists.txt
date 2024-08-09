add_library(textdetect STATIC detect.cpp)

find_package(OpenCV REQUIRED)
list(APPEND TEXTDETECT_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND TEXTDETECT_LIBS ${OpenCV_LIBS})

find_package(Torch REQUIRED)
list(APPEND TEXTDETECT_LIBS ${TORCH_LIBRARIES})

target_include_directories(textdetect PRIVATE ${TEXTDETECT_INCLUDE_DIRS})
target_link_libraries(textdetect PRIVATE ${TEXTDETECT_LIBS})
