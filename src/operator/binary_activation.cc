/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_activation.cc
 * \brief BinaryActivation operator
 * \author Jiaolong Xu
*/
#include "./binary_activation-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BinaryActivationParam param) {
  return new BinaryActivationOp<cpu, mshadow_op::sign, mshadow_op::sign_grad>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryActivationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(BinaryActivationParam);

MXNET_REGISTER_OP_PROPERTY(BinaryActivation, BinaryActivationProp)
.describe("Apply binary activation to input. This is for xnor-net layers.")
.add_argument("data", "Symbol", "Input data to the activation function.")
.add_arguments(BinaryActivationParam::__FIELDS__());
}
}

