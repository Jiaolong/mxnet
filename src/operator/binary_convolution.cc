/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_convolution.cc
 * \brief binary weight convolution
 * \author Jiaolong Xu
*/

#include "./binary_convolution-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BinaryConvolutionParam);

template<>
Operator* CreateOp<cpu>(BinaryConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BinaryConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(BinaryConvolution, BinaryConvolutionProp)
.describe(R"code(Compute *N*-D convolution on *(N+2)*-D input.

In the simplest 2-D convolution, given input data with shape *(batch_size,
channel, height, weight)*, the output is computed by

.. math::

   out[n,i,:,:] = (1/|weight|) * \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
   binarize(weight[i,j,:,:])

where :math:`\star` is the 2-D cross-correlation operator.

For general 2-D convolution, the shapes are

- **data**: *(batch_size, channel, height, weight)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1])*
- **out**: *(batch_size, num_filter, out_height, out_weight)*.

)code" ADD_FILELINE)
.add_argument("data", "ndarray-or-symbol", "Input data to the BinaryConvolutionOp.")
.add_argument("weight", "ndarray-or-symbol", "Weight matrix.")
.add_arguments(BinaryConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
