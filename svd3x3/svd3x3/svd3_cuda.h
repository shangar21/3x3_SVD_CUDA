/**************************************************************************
**
**  svd3
**
**  Quick singular value decomposition as described by:
**  A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
**  Computing the Singular Value Decomposition of 3x3 matrices
**  with minimal branching and elementary floating point operations,
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
**	Identical GPU version
** 	Implementated by: Kui Wu
**	kwu@cs.utah.edu
**
**  May 2018
**
**************************************************************************/

#ifndef SVD3_CUDA_H
#define SVD3_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "math.h" // CUDA math library

#define gone					1065353216
#define gsine_pi_over_eight		1053028117
#define gcosine_pi_over_eight   1064076127
#define gone_half				0.5f
#define gsmall_number			1.e-12f
#define gtiny_number			1.e-20f
#define gfour_gamma_squared		5.8284273147583007813f

union un { float f; unsigned int ui; };
void launch_svd_kernel(float* input, float* u, float*s, float* v);
#endif
