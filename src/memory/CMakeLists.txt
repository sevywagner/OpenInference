add_library(memory STATIC HostArena.cu DeviceArena.cu)
target_include_directories(memory PRIVATE ${INCLUDE_DIR})
