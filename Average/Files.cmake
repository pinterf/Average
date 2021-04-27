FILE(GLOB average_Sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.c"
  "*.cpp"
  "*.h"
)

message("${average_Sources}")

IF( MSVC_IDE )
  # Ninja, unfortunately, seems to have some issues with using rc.exe
  LIST(APPEND masktools_Sources "average.rc")
ENDIF()

