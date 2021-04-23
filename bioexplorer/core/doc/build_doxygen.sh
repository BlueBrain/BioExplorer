#!/bin/sh

doxygen doxygen.cfg
rm -rf ../../../docs/*
mv html/* ../../../docs
