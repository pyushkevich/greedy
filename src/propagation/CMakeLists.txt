message("Configurating Greedy Propagation...")

set(PROPAGATION_LIB_SRC
  PropagationAPI.cxx
  PropagationIO.cxx
  PropagationTools.txx
)

set(PROPAGATION_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GREEDY_SOURCE_DIR}/src
)

add_library(propagationapi ${PROPAGATION_LIB_SRC})
target_link_libraries(propagationapi PUBLIC ${GREEDY_API_LIBS})
target_include_directories(propagationapi PUBLIC ${PROPAGATION_INCLUDE_DIR})

if (BUILD_CLI)
  add_executable(greedy_propagation main.cxx)
  target_link_libraries(greedy_propagation PRIVATE propagationapi)
endif (BUILD_CLI)
