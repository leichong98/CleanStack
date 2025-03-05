add_compiler_rt_component(cleanstack)

set(CLEANSTACK_SOURCES
  cleanstack.cpp
  )

include_directories(..)

set(CLEANSTACK_CFLAGS ${SANITIZER_COMMON_CFLAGS})

foreach(arch ${CLEANSTACK_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.cleanstack
    STATIC
    ARCHS ${arch}
    SOURCES ${CLEANSTACK_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
    CFLAGS ${CLEANSTACK_CFLAGS}
    PARENT_TARGET cleanstack)
endforeach()
