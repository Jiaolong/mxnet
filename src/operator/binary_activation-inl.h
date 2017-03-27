/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_activation.cc
 * \brief BinaryActivation operator
 * \author Jiaolong Xu
*/
#ifndef MXNET_OPERATOR_BINARY_ACTIVATION_INL_H_
#define MXNET_OPERATOR_BINARY_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace binary_activation {
enum BinaryActivationOpInputs {kData};
enum BinaryActivationOpOutputs {kOut};
}

struct BinaryActivationParam : public dmlc::Parameter<BinaryActivationParam> {
  DMLC_DECLARE_PARAMETER(BinaryActivationParam) {}
};

template<typename xpu, typename ForwardOp, typename BackwardOp>
class BinaryActivationOp : public Operator {
  public:
    explicit BinaryActivationOp(BinaryActivationParam p) {}
  
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 2> data = in_data[binary_activation::kData].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> out = out_data[binary_activation::kOut].FlatTo2D<xpu, real_t>(s);
      Assign(out, req[binary_activation::kOut], F<ForwardOp>(data));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(out_grad.size(), 1);
      CHECK(in_data.size() == 1 && in_grad.size() == 1);
      CHECK_EQ(req.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 2> m_out_grad = out_grad[binary_activation::kOut].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> m_out_data = out_data[binary_activation::kOut].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> m_in_grad = in_grad[binary_activation::kData].FlatTo2D<xpu, real_t>(s);
      Assign(m_in_grad, req[binary_activation::kData], F<BackwardOp>(m_out_grad));
  }
}; // class BinaryActivationOp

// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(BinaryActivationParam param);

#if DMLC_USE_CXX11
class BinaryActivationProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(binary_activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
          (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BinaryActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BinaryActivation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[binary_activation::kOut], out_data[binary_activation::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[binary_activation::kOut], in_grad[binary_activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[binary_activation::kData], out_data[binary_activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  BinaryActivationParam param_;
};
#endif  // DMLC_USE_CXX11
} // namespace op
} // namespace mxnet
#endif  // MXNET_OPERATOR_BINARY_ACTIVATION_INL_H_

