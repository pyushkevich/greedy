message("Configurating Greedy Propagation...")

set(PROPAGATION_LIB_SRC
  PropagationAPI.cxx
  PropagationIO.cxx
  PropagationTools.txx
)

add_library(propagationapi ${PROPAGATION_LIB_SRC})
target_link_libraries(propagationapi PUBLIC ${GREEDY_API_LIBS})

if (BUILD_CLI)
  add_executable(greedy_propagation main.cxx)
  target_link_libraries(greedy_propagation PRIVATE propagationapi)
endif (BUILD_CLI)
