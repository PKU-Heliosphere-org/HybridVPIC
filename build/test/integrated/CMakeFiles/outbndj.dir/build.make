# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.0/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jshept/Documents/GitHubOrg/HybridVPIC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jshept/Documents/GitHubOrg/HybridVPIC/build

# Include any dependencies generated for this target.
include test/integrated/CMakeFiles/outbndj.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/integrated/CMakeFiles/outbndj.dir/compiler_depend.make

# Include the progress variables for this target.
include test/integrated/CMakeFiles/outbndj.dir/progress.make

# Include the compile flags for this target's objects.
include test/integrated/CMakeFiles/outbndj.dir/flags.make

test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.o: test/integrated/CMakeFiles/outbndj.dir/flags.make
test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.o: /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/main.cc
test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.o: test/integrated/CMakeFiles/outbndj.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jshept/Documents/GitHubOrg/HybridVPIC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.o"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.o -MF CMakeFiles/outbndj.dir/__/__/deck/main.cc.o.d -o CMakeFiles/outbndj.dir/__/__/deck/main.cc.o -c /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/main.cc

test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/outbndj.dir/__/__/deck/main.cc.i"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/main.cc > CMakeFiles/outbndj.dir/__/__/deck/main.cc.i

test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/outbndj.dir/__/__/deck/main.cc.s"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/main.cc -o CMakeFiles/outbndj.dir/__/__/deck/main.cc.s

test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o: test/integrated/CMakeFiles/outbndj.dir/flags.make
test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o: /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/wrapper.cc
test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o: test/integrated/CMakeFiles/outbndj.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jshept/Documents/GitHubOrg/HybridVPIC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o -MF CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o.d -o CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o -c /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/wrapper.cc

test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.i"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/wrapper.cc > CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.i

test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.s"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jshept/Documents/GitHubOrg/HybridVPIC/deck/wrapper.cc -o CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.s

# Object files for target outbndj
outbndj_OBJECTS = \
"CMakeFiles/outbndj.dir/__/__/deck/main.cc.o" \
"CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o"

# External object files for target outbndj
outbndj_EXTERNAL_OBJECTS =

test/integrated/outbndj: test/integrated/CMakeFiles/outbndj.dir/__/__/deck/main.cc.o
test/integrated/outbndj: test/integrated/CMakeFiles/outbndj.dir/__/__/deck/wrapper.cc.o
test/integrated/outbndj: test/integrated/CMakeFiles/outbndj.dir/build.make
test/integrated/outbndj: libvpic.a
test/integrated/outbndj: /opt/homebrew/Cellar/open-mpi/5.0.3_1/lib/libmpi.dylib
test/integrated/outbndj: test/integrated/CMakeFiles/outbndj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/jshept/Documents/GitHubOrg/HybridVPIC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable outbndj"
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/outbndj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/integrated/CMakeFiles/outbndj.dir/build: test/integrated/outbndj
.PHONY : test/integrated/CMakeFiles/outbndj.dir/build

test/integrated/CMakeFiles/outbndj.dir/clean:
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated && $(CMAKE_COMMAND) -P CMakeFiles/outbndj.dir/cmake_clean.cmake
.PHONY : test/integrated/CMakeFiles/outbndj.dir/clean

test/integrated/CMakeFiles/outbndj.dir/depend:
	cd /Users/jshept/Documents/GitHubOrg/HybridVPIC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jshept/Documents/GitHubOrg/HybridVPIC /Users/jshept/Documents/GitHubOrg/HybridVPIC/test/integrated /Users/jshept/Documents/GitHubOrg/HybridVPIC/build /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated /Users/jshept/Documents/GitHubOrg/HybridVPIC/build/test/integrated/CMakeFiles/outbndj.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : test/integrated/CMakeFiles/outbndj.dir/depend

