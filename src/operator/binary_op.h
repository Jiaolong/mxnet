#ifndef __BINARY_OP__
#define __BINARY_OP__
#include <iostream>
using namespace std;
namespace binary_op {
   
    /*!
     * \brief population count algorithm
     * \tparam n input data
     * \treturn number of bit "1"
     */
    int hacker_popcnt(unsigned int n)
	{
		n -= (n>>1) & 0x55555555;
		n  = (n & 0x33333333) + ((n>>2) & 0x33333333);
		n  = ((n>>4) + n) & 0x0F0F0F0F;
		n += n>>8;
		n += n>>16;
		return n&0x0000003F;
	};

    /*!
     * \brief encode [m, n] size binary data into [m, n/32]
     * \tparam a pointer to the [m, n] size input data
     * \tparam b pointer to the [m, n/32] size output data
     * \tparam m rows
     * \tparam n columns
     */
    void encode_cols(const float* a, unsigned int* b, const int m, const int n) {
        int nn = n / 32 + (n % 32 == 0 ? 0 : 1);
        for(int i = 0; i < m; i++) {
            for (int j = 0; j < nn; j++) {
                unsigned int val = 0;
                unsigned int sign;
                
                for (int k = 0; k < 32; k++) {
                    if (j * 32 + k >= n)
                        sign = 0;
                    else
                        sign = (a[i * n + j * 32 + k] > 0);
                    
                    val |= (sign << k);
                }
                b[i * nn + j] = val;
            }
        } // for i
    };
   
    /*!
     * \brief encode [m, n] size binary maxtrix into [m/32, n] size
     * \tparam a pointer to input binary matrix
     * \tparam b pointer to output maxtrix
     * \tparam m rows
     * \tparam n columns
     */
    void encode_rows(const float* a, unsigned int* b, const int m, const int n) {
        int mm = m / 32 + (m % 32 == 0 ? 0 : 1);
        for(int j = 0; j < n; j++) {
            for (int i = 0; i < mm; i++) {
                unsigned int val = 0;
                unsigned int sign;
                
                for (int k = 0; k < 32; k++) {
                    if (i * 32 + k >= m)
                        sign = 0;
                    else
                        sign = (a[j + (i * 32 + k) * n] > 0);
                    
                    val |= (sign << k);
                }
                b[j + i * n] = val;
            }
        } // for j
    };

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
            const int m, const int n, const int k, float* out, const float* alpha) {
        int a = n % 32;
        int nn = n / 32 + (a == 0 ? 0 : 1);
        int total_bits = (nn - 1) * 32 + (a == 0 ? 32 : a);
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                out[i * k + p] = 0;
                for (int j = 0; j < nn; j++) {
                    out[i * k + p] += float(hacker_popcnt(WB[i * nn + j] ^ AB[j * k + p]));
                }
                out[i * k + p] = alpha[i] * (total_bits - 2 * out[i * k + p]);
            }
        }
    };

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
            const int m, const int n, const int k, float* out, const float* alpha) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < k; j++) {
                float c = 0.0;
                for (size_t p = 0; p < n; p++) {
                    if (W[i * n + p] > 0)
                        c += A[p * k + j];
                    else if (W[i * n + p] < 0)
                        c -= A[p * k + j];
                }
                
                out[i * k + j] = c * alpha[i];
            }
        }
  };
}
#endif // __BINARY_OP__
