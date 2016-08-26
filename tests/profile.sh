#!/bin/bash
# Script to profile Python2.7 functions and methods. Use decorator @profile in the file on the function to be tested.
# Remember to use dos2unix on this script if it has been edited in Windows.

directory=prof_tests
if [ ! -d $directory ]; then
  mkdir $directory
fi

echo Profiling...this could take a while.

# Generate a filename with the Python file run and the start time.
cmds=$@
start_time=`date '+%Y_%m_%d_%H_%M_%S'`;
set -- $cmds
filename=$1
filepath=$directory/$filename-$start_time.prof

# Currently all print statements from the Python file are logged, followed by memory usage and runtime.
echo Profiling memory usage.
python -m memory_profiler $cmds > $filepath
echo Memory usage profiled.

echo Profiling runtime.
kernprof -l $cmds 
# Append runtime results to same file as memory results.
python -m line_profiler $filename.lprof >> $filepath 
echo Runtime profiled. All results saved in $filepath.
rm $filename.lprof
echo Deleted $filename.lprof.