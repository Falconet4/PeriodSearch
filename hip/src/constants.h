#pragma once
#define POINTS_MAX         20000             /* max number of data points in one lc. */
/* Curve2 staging width: sh_dyda[TILE][C2S] in curve2_CUDA.hip uses a local
   C2S = 67; the host guard below must agree with it. */
#define C2S_TILE           67
#define MAX_N_OBS         20000             /* max number of data points */
#define MAX_LC              200             /* max number of lightcurves */
#define MAX_LINE_LENGTH    1000             /* max length of line in an input file */
#define MAX_N_FAC          1000             /* max number of facets */
#define MAX_N_ITER          100             /* maximum number of iterations */
#define MAX_N_PAR           200             /* maximum number of parameters */
#define MAX_LM               10             /* maximum degree and order of sph. harm. */
#define N_PHOT_PAR            5             /* maximum number of parameters in scattering  law */
#define TINY                  1e-8          /* precision parameter for mu, mu0*/
#define N_POLES              10             /* number of initial poles */

#define PI                 M_PI             /* 3.14159265358979323846 */
#define AU            149597870.691         /* Astronomical Unit [km] */
#define C_SPEED       299792458             /* speed of light [m/s]*/

#define DEG2RAD      (PI / 180)
#define RAD2DEG      (180 / PI)

//MUST BE 128 or 64

#define CUDA_BLOCK_DIM 128
#define CURVE1LAST_BLOCK_DIM 64

// Curve2 uses two waves per block; only the parameter columns are active.
#define CURVE2_BLOCK_DIM 64