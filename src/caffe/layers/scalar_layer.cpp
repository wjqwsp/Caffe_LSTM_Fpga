#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScalarLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // TODO: make ScalarLayer usable in-place.
  // Currently, in-place computation is broken during Backward with
  // propagate_down[0] && propagate_down[1], as bottom[0]'s diff is used for
  // temporary storage of an intermediate result, overwriting top[0]'s diff
  // if using in-place computation.
  CHECK_NE(bottom[0], top[0]) << "ScalarLayer cannot be used in-place";
  axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.scalar_param().axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + bottom[1]->num_axes())
      << "bottom[1]'s shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), bottom[1]->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and bottom[1]->shape(" << i << ")";
  }
    // bottom[0]的数目为outer_dim * scalar_dim * inner_dim
    // bottom[1]的数目为scalar_dim
  outer_dim_ = bottom[0]->count(0, axis_);     // 外层数据份数
  scalar_dim_ = bottom[1]->count();    // 对于每一份外层数据，一共的需要缩放的系数数目
  inner_dim_ = bottom[0]->count(axis_ + bottom[1]->num_axes());    // 被同一系数缩放的数据量
  top[0]->ReshapeLike(*bottom[0]);    // top和bottom的维度应完全一样
  sum_result_.Reshape(vector<int>(1, outer_dim_ * scalar_dim_));  // 用于计算bottom[1]的反向误差，作为中间值求和以降维
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);    // 用于计算bottom[1]的反向误差，用作对矩阵某维度求和
  sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)) {
    caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ScalarLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {    // 简单地进行数据缩放
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scalar_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scalar_dim_; ++d) {
      const Dtype factor = scalar_data[d];
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
}

template <typename Dtype>
void ScalarLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scalar diff, and we're done.
    const bool is_eltwise = (inner_dim_ == 1 && outer_dim_ == 1);
    Dtype* product = is_eltwise ?
        bottom[1]->mutable_cpu_diff() : bottom[0]->mutable_cpu_diff();
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);    // 如果outer_dim和inner_dim均为1，说明计算缩放相当于两个向量点乘，结果恰好是向量，那么算反向误差也直接点乘就行
    if (!is_eltwise) {   // 运行到这说明inner_dim或outer_dim至少一个不为1，需要用bottom[0]的误差矩阵提供较大的缓存空间暂存反向误差。后面就是要将outer_dim和inner_dim这两个维度通过求和降掉
      Dtype* sum_result = NULL;
      if (inner_dim_ == 1) {    // inner_dim为1，说明结果刚好能放进sum_result
        sum_result = product;
      } else if (sum_result_.count() == 1) {  // 这里说明scalar_dim和outer_dim均为1，将inner_dim求和后便是最终结果
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scalar_diff = bottom[1]->mutable_cpu_diff();
        *scalar_diff = caffe_cpu_dot(inner_dim_, product, sum_mult);
      } else {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        sum_result = (outer_dim_ == 1) ?    // 如果outer_dim为1，则对inner_dim求和之后便得到scalar_dim长的向量作为最终结果
            bottom[1]->mutable_cpu_diff() : sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,    // 如果outer_dim不为1，则求和后仍需暂存到sum_result中
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (outer_dim_ != 1) {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scalar_diff = bottom[1]->mutable_cpu_diff();
        if (scalar_dim_ == 1) {
          *scalar_diff = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);    // scalar_dim为1，则最终结果是一个标量，向量点乘即可
        } else {
          caffe_cpu_gemv(CblasTrans, outer_dim_, scalar_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(0), scalar_diff);    // 这是一般情况，将sum_result转置后乘上全为1的向量进行求和降维
        }
      }
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scalar_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < outer_dim_; ++n) {
      for (int d = 0; d < scalar_dim_; ++d) {
        const Dtype factor = scalar_data[d];
        caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);    // bottom[0]的反向误差计算方法与正向传播基本一致
        bottom_diff += inner_dim_;
        top_diff += inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScalarLayer);
#endif

INSTANTIATE_CLASS(ScalarLayer);
REGISTER_LAYER_CLASS(Scalar);

}  // namespace caffe
