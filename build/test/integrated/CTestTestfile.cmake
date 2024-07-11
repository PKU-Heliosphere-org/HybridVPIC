# CMake generated Testfile for 
# Source directory: /Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated
# Build directory: /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(accel "/opt/homebrew/bin/mpiexec" "-n" "1" "accel" "1 1")
set_tests_properties(accel PROPERTIES  _BACKTRACE_TRIPLES "/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;9;add_test;/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;0;")
add_test(cyclo "/opt/homebrew/bin/mpiexec" "-n" "1" "cyclo" "1 1")
set_tests_properties(cyclo PROPERTIES  _BACKTRACE_TRIPLES "/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;9;add_test;/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;0;")
add_test(inbndj "/opt/homebrew/bin/mpiexec" "-n" "1" "inbndj" "1 1")
set_tests_properties(inbndj PROPERTIES  _BACKTRACE_TRIPLES "/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;9;add_test;/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;0;")
add_test(interpe "/opt/homebrew/bin/mpiexec" "-n" "1" "interpe" "1 1")
set_tests_properties(interpe PROPERTIES  _BACKTRACE_TRIPLES "/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;9;add_test;/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;0;")
add_test(outbndj "/opt/homebrew/bin/mpiexec" "-n" "1" "outbndj" "1 1")
set_tests_properties(outbndj PROPERTIES  _BACKTRACE_TRIPLES "/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;9;add_test;/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;0;")
add_test(pcomm "/opt/homebrew/bin/mpiexec" "-n" "8" "pcomm" "1 1")
set_tests_properties(pcomm PROPERTIES  _BACKTRACE_TRIPLES "/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;12;add_test;/Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated/CMakeLists.txt;0;")
