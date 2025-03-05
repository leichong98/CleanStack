set(CLEANSTACK_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLEANSTACK_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CLEANSTACK_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND CLEANSTACK_TEST_DEPS cleanstack)

  # Some tests require LTO, so add a dependency on the relevant LTO plugin.
  if(LLVM_ENABLE_PIC)
    if(LLVM_BINUTILS_INCDIR)
      list(APPEND CLEANSTACK_TEST_DEPS
        LLVMgold
      )
    endif()
    if(APPLE)
      list(APPEND CLEANSTACK_TEST_DEPS
        LTO
      )
    endif()
  endif()
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  )

add_lit_testsuite(check-cleanstack "Running the CleanStack tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLEANSTACK_TEST_DEPS})
set_target_properties(check-cleanstack PROPERTIES FOLDER "Compiler-RT Misc")