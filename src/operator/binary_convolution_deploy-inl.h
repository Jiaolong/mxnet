/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_convolution_deploy-inl.h
 * \brief binary weight convolution for deployment
 * \author Jiaolong Xu
*/
#ifndef MXNET_OPERATOR_BINARY_CONVOLUTION_DEPLOY_INL_H_
#define MXNET_OPERATOR_BINARY_CONVOLUTION_DEPLOY_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./binary_op.h"

namespace mxnet {
namespace op {

namespace conv {
enum ConvolutionOpInputs {kData};
enum ConvolutionOpAuxiliary {kBinWeight, kAlpha};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
}

struct BinaryConvolutionDeployParam : public dmlc::Parameter<BinaryConvolutionDeployParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_group;
  uint32_t num_filter;
  uint64_t workspace;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(BinaryConvolutionDeployParam) {
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("convolution stride: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for convolution: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temperal workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCHW for 2d and NCDHW for 3d.");
  }
};

template<typename xpu, typename DType>
class BinaryConvolutionDeployOp : public Operator {
 public:
  explicit BinaryConvolutionDeployOp(BinaryConvolutionDeployParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCHW) 
        << "Only support NCHW layout";
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    
    CHECK_EQ(aux_args.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data  = in_data[conv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out   = out_data[conv::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> alpha = aux_args[conv::kAlpha].get<xpu, 1, DType>(s);

    const index_t nbatch  = data.shape_[0];
    const index_t input_c = data.shape_[1];
    const index_t input_h = data.shape_[2];
    const index_t input_w = data.shape_[3];
    const index_t f_n = param_.num_filter;
    const index_t f_h = param_.kernel[0];
    const index_t f_w = param_.kernel[1];
    const index_t out_h = out.shape_[2];
    const index_t out_w = out.shape_[3];

    Shape<2> w_shape = Shape2(f_n, input_c * f_h * f_w);
    // length of encoded array
    int len_encode = w_shape[1] / 32 + (w_shape[1] % 32 == 0 ? 0 : 1); 
    index_t temp_size = this->InitTemp(data.shape_, out.shape_);
    temp_size += len_encode * out_w * out_h * nstep_; 
    Tensor<xpu, 1, DType> workspace 
        = ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
                Shape1(temp_size), s);
 
    // encoded binary weight
    Tensor<xpu, 2, DType> w_bin = aux_args[conv::kBinWeight].get<xpu, 2, DType>(s);

    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      // storage of image to column data, [f_n * f_h * f_w, out_h * out_w * step]
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
            workspace.dptr_, Shape2(w_shape[1], out_h * out_w * step), s);
    
      // storage of dot product result of a path and filter
      Tensor<xpu, 2, DType> temp_dst = Tensor<xpu, 2, DType>(
            workspace.dptr_ + temp_col.shape_.Size(), Shape2(f_n, out_h * out_w * step), s);


      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step),
                                    param_.pad[0], param_.pad[1]),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      }

      // temporary memory of encoded binary data
      Tensor<xpu, 2, DType> temp_col_bin = Tensor<xpu, 2, DType>(
            workspace.dptr_ + temp_col.shape_.Size() + temp_dst.shape_.Size(),
            Shape2(len_encode, out_h * out_w * step), s);
      
      // encode rows [n, k] -> [n/32, k]
      binary_op::encode_rows((float*) temp_col.dptr_, (unsigned int*) temp_col_bin.dptr_,
            temp_col.size(0), temp_col.size(1));

      binary_op::popcount_xnor_dot((unsigned int*) w_bin.dptr_,
            (unsigned int*) temp_col_bin.dptr_,
            w_shape[0], w_shape[1], temp_col.size(1),
            (float*)temp_dst.dptr_, (float*)alpha.dptr_); 
      
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                              mshadow::Shape4(param_.num_filter,
                                                  step,
                                                  out.size(2),
                                                  out.size(3))));
    }
  }
 
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {}

 private: 
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape2(param_.num_filter,
                                     oshape[2] * oshape[3]);
    // param_.workspace is in elements of sizeof(DType)
    // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
    nstep_ = std::max(
        std::min(
            static_cast<index_t>(
                param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
            ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<2> sdst = mshadow::Shape2(shape_dstunit_[0],
                                             shape_dstunit_[1] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(DType) << " Bytes";
    return required_size;
  }
  
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<2> shape_dstunit_;
  index_t nstep_;
  BinaryConvolutionDeployParam param_;
};  // class BinaryConvolutionDeployOp

template<typename xpu>
Operator* CreateOp(BinaryConvolutionDeployParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class BinaryConvolutionDeployProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "weight"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3) << param_.kernel.ndim() << "D convolution not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    out_shape->resize(1, TShape());
    aux_shape->resize(2, TShape());
    const TShape &dshp = (*in_shape)[conv::kData];
    if (dshp.ndim() ==  0) return false;
    CHECK_EQ(param_.kernel.ndim(), 2);

    // 2d conv
    CHECK_EQ(dshp.ndim(), 4) \
      << "Input data should be 4D in batch-num_filter-y-x";
    Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
    Shape<4> wshape = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
                           param_.kernel[0], param_.kernel[1]);

    int w_len = wshape[1] * wshape[2] * wshape[3];
    int len_encode = w_len / 32 + (w_len % 32 == 0 ? 0 : 1); 
    Shape<2> wbinshape = Shape2(param_.num_filter, len_encode);
    Shape<1> alpha_shape = Shape1(param_.num_filter);

    SHAPE_ASSIGN_CHECK(*aux_shape, conv::kBinWeight, wbinshape);
    SHAPE_ASSIGN_CHECK(*aux_shape, conv::kAlpha, alpha_shape);

    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<4> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = (dshape[2] + 2 * param_.pad[0] -
      (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
    oshape[3] = (dshape[3] + 2 * param_.pad[1] -
      (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
    dshape[0] = oshape[0];
    if (param_.stride[0] == 1) {
    dshape[2] = oshape[2] + param_.dilate[0] * (ksize_y - 1) - 2 * param_.pad[0];
    }
    if (param_.stride[1] == 1) {
    dshape[3] = oshape[3] + param_.dilate[1] * (ksize_x - 1) - 2 * param_.pad[1];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
                      ConvertLayout(dshape, kNCHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
    CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
    }
    if (dshape[3] != 0) {
    CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
    }
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

    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype);

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BinaryConvolutionDeployProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BinaryConvolutionDeploy";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[conv::kOut], in_data[conv::kData]};
  }

  std::vector<ResourceRequest> ForwardResource(
          const std::vector<TShape> &in_shape) const override {
      return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"binary_weight", "alpha"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BinaryConvolutionDeployParam param_;
};  // class BinaryConvolutionDeployProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BINARY_CONVOLUTION_DEPLOY_INL_H_
