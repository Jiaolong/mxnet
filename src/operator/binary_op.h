/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_op.h
 * \brief binary operation
 * \author Jiaolong Xu
*/
#ifndef __BINARY_OP__
#define __BINARY_OP__
namespace binary_op {
   
    /*!
     * \brief population count algorithm
     * \tparam n input data
     * \treturn number of bit "1"
     */
    int hacker_popcnt(unsigned int n);
	
    /*!
     * \brief encode [m, n] size binary data into [m, n/32]
     * \tparam a pointer to the [m, n] size input data
     * \tparam b pointer to the [m, n/32] size output data
     * \tparam m rows
     * \tparam n columns
     */
    void encode_cols(const float* a, unsigned int* b, const int m, const int n);
           
    /*!
     * \brief encode [m, n] size binary maxtrix into [m/32, n] size
     * \tparam a pointer to input binary matrix
     * \tparam b pointer to output maxtrix
     * \tparam m rows
     * \tparam n columns
     */
    void encode_rows(const float* a, unsigned int* b, const int m, const int n);
        
    /*!
     * \brief compute dot product of encoded binary arrays by popcount(xnor)
     * \tparam WB pointer of encoded weight with size [m, n/32]
     * \tparam AB pointer of encoded data with size [n/32, k]
     * \tparam m rows of WB
     * \tparam n columns of W
     * \tparam k columns of AB
     * \tparam alpha coefficients of each filter in WB, with shape [m,]
     * \tparam out pointer to the dot product result with size [m, k]
     */ 
    void popcount_xnor_dot(const unsigned int* WB, const unsigned int* AB,
            const int m, const int n, const int k, float* out, const float* alpha);
        
    /*!
     * \brief compute dot product of binary weight and real-value data [row major]
     * \tparam WB pointer of the binary weight with size [m, n]
     * \tparam AB pointer of data with size [n, k]
     * \tparam m rows of W
     * \tparam n columns of W
     * \tparam k columns of A
     * \tparam alpha coefficients of each filter in W, with shape [m,]
     * \tparam out pointer to the dot product result with size [m, k]
     */ 
    void bw_dot(const float* W, const float* A, 
            const int m, const int n, const int k, float* out, const float* alpha);
#endif // __BINARY_OP__
