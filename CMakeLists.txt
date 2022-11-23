cmake_minimum_required(VERSION 3.5)

project(Parcial3HPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Se crea una variable para las fuentes
set(project_sources
    main.cpp
    ClassExtraction/extractiondata.cpp
    Regression/regression.cpp)

#Se crea una variable para las interfaces
set(project_headers
    main.cpp
    ClassExtraction/extractiondata.h
    Regression/regression.h)

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})
