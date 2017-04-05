/*!
 * Copyright (c) 2017 by Contributors
 * \file binary_convolution-inl.h
 * \brief binary weight convolution
 * \author Jiaolong Xu
*/
#ifndef MXNET_OPERATOR_BINARY_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_BINARY_CONVOLUTION_INL_H_

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
enum ConvolutionOpInputs {kData, kWeight};
enum ConvolutionOpAuxiliary {kBinWeight};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
}

struct BinaryConvolutionParam : public dmlc::Parameter<BinaryConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_group;
  uint32_t num_filter;
  bool xnor_net;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(BinaryConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("convolution stride: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for convolution: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(xnor_net).set_default(false)
    .describe("wether or not xnor network.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
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
class BinaryConvolutionOp : public Operator {
 public:
  explicit BinaryConvolutionOp(BinaryConvolutionParam p) {
    this->param_ = p;
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
    
    CHECK_EQ(aux_args.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data  = in_data[conv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out   = out_data[conv::kOut].get<xpu, 4, DType>(s);
    const size_t nbatch  = data.shape_[0];
    const size_t input_c = data.shape_[1];
    const size_t input_h = data.shape_[2];
    const size_t input_w = data.shape_[3];
    const size_t f_n = param_.num_filter;
    const size_t f_h = param_.kernel[0];
    const size_t f_w = param_.kernel[1];
    const size_t out_h = out.shape_[2];
    const size_t out_w = out.shape_[3];

    Shape<2> w_shape = Shape2(f_n, input_c * f_h * f_w);
    // length of encoded array
    int len_encode = w_shape[1] / 32 + (w_shape[1] % 32 == 0 ? 0 : 1); 

    Tensor<xpu, 2, DType> w_bin;
    if (param_.xnor_net) {
        w_bin = aux_args[conv::kBinWeight].get<xpu, 2, DType>(s);
        CHECK_EQ(w_bin.shape_[0], f_n) << "binary weight shape[0]";
        CHECK_EQ(w_bin.shape_[1], len_encode) << "binary weight shape[1]";
    }

    Tensor<xpu, 2, DType> w_real =
        in_data[conv::kWeight].get_with_shape<xpu, 2, DType>(w_shape, s);
     
    // compute alpha
    size_t num_elements = w_shape[1];
    alpha_.resize(f_n);
    
    for (index_t j = 0; j < w_real.size(0); j++) {
        alpha_[j] = 0.0;

        for (index_t k = 0; k < w_real.size(1); k++)
          alpha_[j] += fabsf(float(w_real[j][k]));
        
        alpha_[j] /= num_elements;
    }
     
    int temp_size = w_shape[1] * out_h * out_w + f_n * out_h * out_w;
    if (param_.xnor_net)
        temp_size += len_encode * out_w * out_h;
    
    Tensor<xpu, 1, DType> workspace 
        = ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
                Shape1(temp_size), s);

    // storage of image to column data
    Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
            workspace.dptr_, Shape2(w_shape[1], out_h * out_w), s);
    // storage of dot product result of a path and filter
    Tensor<xpu, 2, DType> temp_dst = Tensor<xpu, 2, DType>(
            workspace.dptr_ + temp_col.shape_.Size(), Shape2(f_n, out_h * out_w), s);

    Tensor<xpu, 2, DType> temp_col_bin; 
    if (param_.xnor_net) { 
        // storage of encoded binary data
        temp_col_bin = Tensor<xpu, 2, DType>(
                workspace.dptr_ + temp_col.shape_.Size() + temp_dst.shape_.Size(),
                Shape2(len_encode, out_h * out_w), s);
       
        // encode weight [m, n] -> [m, n/32]
        binary_op::encode_cols((float*) w_real.dptr_, (unsigned int*) w_bin.dptr_,
               w_real.size(0), w_real.size(1));
    }
    
    const index_t step = 1;
    for (index_t i = 0; i < nbatch; i += step) {
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

      if (param_.xnor_net) {
          // encode rows [n, k] -> [n/32, k]
          binary_op::encode_rows((float*) temp_col.dptr_, (unsigned int*) temp_col_bin.dptr_,
                  temp_col.size(0), temp_col.size(1));

          binary_op::popcount_xnor_dot((unsigned int*) w_bin.dptr_,
                  (unsigned int*) temp_col_bin.dptr_,
                  w_real.size(0), w_real.size(1), temp_col.size(1),
                  (float*)temp_dst.dptr_, alpha_.data());
      } else {
          // dot product
          binary_op::bw_dot((float*)w_real.dptr_, (float*)temp_col.dptr_, 
                  w_real.size(0), w_real.size(1), temp_col.size(1),
                  (float*)temp_dst.dptr_, alpha_.data());
      }
      
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
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[conv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[conv::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[conv::kData].get<xpu, 4, DType>(s);
 
    const size_t nbatch  = data.shape_[0];
    const size_t input_c = data.shape_[1];
    const size_t input_h = data.shape_[2];
    const size_t input_w = data.shape_[3];
    const size_t f_n = param_.num_filter;
    const size_t f_h = param_.kernel[0];
    const size_t f_w = param_.kernel[1];
    const size_t out_h = grad.shape_[2];
    const size_t out_w = grad.shape_[3];

    Shape<2> w_shape = Shape2(f_n, input_c * f_h * f_w);
    Tensor<xpu, 2, DType> w_real =
        in_data[conv::kWeight].get_with_shape<xpu, 2, DType>(w_shape, s);
    Tensor<xpu, 2, DType> gwmat =
        in_grad[conv::kWeight].get_with_shape<xpu, 2, DType>(w_shape, s);
        
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(input_c * f_h * f_w * out_h * out_w + f_n * out_h * out_w + w_shape[1] * w_shape[0]), s);

    
    Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(workspace.dptr_, 
            Shape2(input_c * f_h * f_w, out_h * out_w), s);
    
    Tensor<xpu, 2, DType> temp_dst = Tensor<xpu, 2, DType>(
            workspace.dptr_ + temp_col.shape_.Size(),
            Shape2(f_n, out_h * out_w), s);
    
    Tensor<xpu, 2, DType> w_bin = Tensor<xpu, 2, DType>(
            workspace.dptr_ + temp_col.shape_.Size() + temp_dst.shape_.Size(),
            Shape2(w_shape[0], w_shape[1]), s);

    for (index_t i = 0; i < w_shape[0]; i++) {
        for (index_t j = 0; j < w_shape[1]; j++) {
            if (w_real[i][j] > 0)
                w_bin[i][j] = alpha_[i];
            else if (w_real[i][j] < 0)
                w_bin[i][j] = -1.0 * alpha_[i];
            else
                w_bin[i][j] = 0;
        }
    }

    const index_t step = 1;
    for (index_t i = 0; i < nbatch; i += 1) {

      temp_dst = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      }
      
      if (i == 0) {
        Tensor<xpu, 2, DType> tmp_gwmat = gwmat;
        Assign(tmp_gwmat, req[conv::kWeight], dot(temp_dst, temp_col.T()));
      } else {
        gwmat += dot(temp_dst, temp_col.T());
      }

      //binary_op::bw_dot((float*) w_real_t.dptr_, (float*) temp_dst.dptr_, 
      //        w_real.size(1), w_real.size(0), temp_dst.size(1),
      //        (float*) temp_col.dptr_, alpha_.data());
      
      temp_col = dot(w_bin.T(), temp_dst);
      
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        Assign(gdata.Slice(i, i + step), req[conv::kData],
               pack_col2patch(temp_col,
                              data.Slice(i, i + step).shape_,
                              param_.kernel[0],
                              param_.kernel[1],
                              param_.stride[0],
                              param_.stride[1],
                              param_.dilate[0],
                              param_.dilate[1]));
      } else {
        Shape<4> pshape = data.Slice(i, i + step).shape_;
        pshape[2] += 2 * param_.pad[0];
        pshape[3] += 2 * param_.pad[1];
        Assign(gdata.Slice(i, i + step), req[conv::kData],
               crop(pack_col2patch(temp_col,
                                   pshape,
                                   param_.kernel[0],
                                   param_.kernel[1],
                                   param_.stride[0],
                                   param_.stride[1],
                                   param_.dilate[0],
                                   param_.dilate[1]),
                    gdata[i][0].shape_));
      }
    }
    // update gradient
    float ee = 1.0 / (w_shape[1]);
    
    for (index_t j = 0; j < w_real.size(0); j++)
        for (index_t k = 0; k < w_real.size(1); k++)
            if (w_real[j][k] >= 1 || w_real[j][k] <= -1)
                gwmat[j][k] *= ee;
            else
                gwmat[j][k] *= (alpha_[j] * w_real[j][k] + ee);
  
  }

 private: 
  BinaryConvolutionParam param_;
  std::vector<float> alpha_;
};  // class BinaryConvolutionOp

template<typename xpu>
Operator* CreateOp(BinaryConvolutionParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class BinaryConvolutionProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    out_shape->resize(1, TShape());
    aux_shape->resize(1, TShape());
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

    wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    SHAPE_ASSIGN_CHECK(*aux_shape, conv::kBinWeight, wbinshape);

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
    auto ptr = new BinaryConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BinaryConvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[conv::kOut], in_data[conv::kData], in_data[conv::kWeight]};
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
    return {"binary_weight"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BinaryConvolutionParam param_;
};  // class BinaryConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BINARY_CONVOLUTION_INL_H_
