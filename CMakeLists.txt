project(bundleAdjustment)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

#aux_source_directory(. dir_srcs) #aux_source_directory(<dir> <variable>)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( G2O REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )

include_directories( ${PROJECT_SOURCE_DIR} ${G2O_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} )

target_link_libraries( ${PROJECT_NAME}
        ${OpenCV_LIBS}
        g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
        ${CHOLMOD_LIBRARIES}
        ) #libraries




