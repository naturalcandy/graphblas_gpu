#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include <graphblas_gpu/kernels/ewise_ops.hpp>
#include <graphblas_gpu/kernels/spmv_csr.hpp>
#include <graphblas_gpu/kernels/spmv_ell.hpp>
#include <graphblas_gpu/kernels/spmv_sellc.hpp>

#include "ewise_ops.cu"     
#include "spmv_csr.cu"
#include "spmv_ell.cu"
#include "spmv_sellc.cu"
