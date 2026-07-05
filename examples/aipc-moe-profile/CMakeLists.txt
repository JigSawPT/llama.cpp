set(TARGET llama-aipc-moe-profile)
add_executable(${TARGET} aipc-moe-profile.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# aipc-ffn-diff: single-layer FFN tensor capture for sign-sensitive dual-chain verification
set(TARGET llama-aipc-ffn-diff)
add_executable(${TARGET} aipc-ffn-diff.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
