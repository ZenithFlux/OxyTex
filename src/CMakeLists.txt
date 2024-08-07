add_executable(oxytex oxytex.cpp)
list(APPEND OXYTEX_INCLUDE_DIRS ".")

find_package(OpenCV REQUIRED)
list(APPEND OXYTEX_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND OXYTEX_LIBS ${OpenCV_LIBS})

add_subdirectory(textdetect)
list(APPEND OXYTEX_LIBS textdetect)

target_include_directories(oxytex PRIVATE ${OXYTEX_INCLUDE_DIRS})
target_link_libraries(oxytex PRIVATE ${OXYTEX_LIBS})
