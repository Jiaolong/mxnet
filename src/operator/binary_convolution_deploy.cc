/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_convolution_deploy.cc
 * \brief binary weight convolution in deploy mode
 * \author Jiaolong Xu
*/

#include "./binary_convolution_deploy-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BinaryConvolutionDeployParam);

template<>
Operator* CreateOp<cpu>(BinaryConvolutionDeployParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BinaryConvolutionDeployOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryConvolutionDeployProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(BinaryConvolutionDeploy, BinaryConvolutionDeployProp)
.describe(R"code(Compute *N*-D binary weight convolution on *(N+2)*-D binary input.
)code" ADD_FILELINE)
.add_argument("data", "ndarray-or-symbol", "Input data to the BinaryConvolutionDeployOp.")
.add_arguments(BinaryConvolutionDeployParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
