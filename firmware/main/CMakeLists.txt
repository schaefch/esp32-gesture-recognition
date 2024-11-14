
#
# Main component of TF Micro project 'micro_speech'.
#

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
idf_component_register(
    SRCS main.cc main_functions.cc output_handler.cc 
    PRIV_REQUIRES spi_flash driver 
    INCLUDE_DIRS "")
