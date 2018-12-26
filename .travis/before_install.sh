#!/bin/bash

export NUMCORES=`grep -c ^processor /proc/cpuinfo`
if [ ! -n "$NUMCORES" ]; then
  export NUMCORES=`sysctl -n hw.ncpu`
fi
echo Using $NUMCORES cores

sudo apt update
sudo apt install cmake libprotobuf-dev protobuf-compiler
sudo apt install python-pip python-opencv python-numpy
sudo apt remove python-enum34 --purge
sudo pip install tensorflow==1.12.0
