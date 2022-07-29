#!/bin/bash
# Compile cpp subsampling
cd point-transformer/cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ../..
# # Compile cuda point ops
cd pt_custom_ops/
python3 setup.py build_ext --inplace
