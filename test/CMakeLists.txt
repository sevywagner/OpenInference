if (INSTALL_GTEST)
	include(FetchContent)
	FetchContent_Declare(google_test
		GIT_REPOSITORY https://github.com/google/googletest
		GIT_TAG v1.18.0)
	
	FetchContent_GetProperties(google_test)
	if (NOT google_test_POPULATED)
		FetchContent_MakeAvailable(google_test)
	endif()
endif()

add_executable(test test.cu)
target_link_libraries(test PRIVATE gtest_main gmock_main inference)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
