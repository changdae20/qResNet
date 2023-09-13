################## OPENCV #####################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
###############################################

################# PROTOBUF #################
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
############################################

################## fmt #################
find_package(fmt REQUIRED)
########################################

################  MPFR  ################
include_directories( /usr/lib/GMP/include )
include_directories( /usr/lib/MPFR/include )
link_directories( /usr/lib/GMP/lib )
link_directories( /usr/lib/MPFR/lib )
########################################

SET(PROTOBUF_SRC_FILES
    onnx.proto3.pb.cc
    onnx.proto3.pb.h
)

link_directories(${PROJECT_NAME})

add_executable(main
	${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/layers.hpp
    ${PROTOBUF_SRC_FILES}
    ${PROJECT_SOURCE_DIR}/src/AFloat.hpp
    ${PROJECT_SOURCE_DIR}/src/threadpool.hpp
    ${PROJECT_SOURCE_DIR}/src/CatGenerator.hpp
    ${PROJECT_SOURCE_DIR}/src/util.hpp
    ${PROJECT_SOURCE_DIR}/src/onnx.hpp
)

target_link_libraries(main
    ${OpenCV_LIBS}
    ${PROTOBUF_LIBRARIES}
    fmt
    gmp
    mpfr
)