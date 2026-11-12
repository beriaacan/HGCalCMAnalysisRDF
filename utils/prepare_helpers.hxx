#ifndef HGCAL_CM_PREPARE_HELPERS_HXX
#define HGCAL_CM_PREPARE_HELPERS_HXX
#include <ROOT/RVec.hxx>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace hgcal_cm_prepare {
using ROOT::VecOps::RVec;
inline float missing() { return std::numeric_limits<float>::quiet_NaN(); }
template<class C> bool present(const RVec<C>& channels, int nch) {
  std::vector<bool> seen(nch, false);
  for (auto value : channels) {
    int ch = int(value);
    if (ch < 0 || ch >= nch) throw std::runtime_error("Out-of-range module channel");
    if (seen[ch]) throw std::runtime_error("Duplicate module channel");
    seen[ch] = true;
  }
  return !channels.empty();
}
template<class C, class V> RVec<float> dense(const RVec<C>& channels, const RVec<V>& values, int nch) {
  if (channels.size() != values.size()) throw std::runtime_error("Channel/value size mismatch");
  RVec<float> output(nch, missing());
  for (size_t i = 0; i < channels.size(); ++i) {
    int ch = int(channels[i]);
    if (ch < 0 || ch >= nch) throw std::runtime_error("Out-of-range selected channel");
    output[ch] = float(values[i]);
  }
  return output;
}
template<class C, class V> RVec<float> erx_cm(const RVec<C>& channels, const RVec<V>& sums, int nerx) {
  if (channels.size() != sums.size()) throw std::runtime_error("CM/channel size mismatch");
  RVec<float> output(nerx, missing());
  for (size_t i = 0; i < channels.size(); ++i) {
    int ch = int(channels[i]);
    if (ch < 0 || ch >= 37*nerx) throw std::runtime_error("CM channel out of range");
    int erx = ch / 37;
    float value = 0.5f * float(sums[i]);
    if (std::isfinite(output[erx]) && output[erx] != value)
      throw std::runtime_error("Inconsistent HGCDigi_cm within one module/eRx");
    output[erx] = value;
  }
  return output;
}
inline RVec<float> expand_cm(const RVec<float>& cm) {
  RVec<float> output(37*cm.size(), missing());
  for (size_t ch = 0; ch < output.size(); ++ch) output[ch] = cm[ch/37];
  return output;
}
inline RVec<float> mask_absent(RVec<float> values, const RVec<float>& absent) {
  if (values.size()!=37*absent.size()) throw std::runtime_error("Absent eRx mask size mismatch");
  for (size_t ch=0; ch<values.size(); ++ch) if (absent[ch/37]) values[ch]=missing();
  return values;
}
inline unsigned count_finite(const RVec<float>& values) {
  unsigned result=0;
  for (auto value:values) if (std::isfinite(value)) ++result;
  return result;
}
inline double sum_finite(const RVec<float>& values) {
  double total = 0.;
  for (auto value : values) if (std::isfinite(value)) total += value;
  return total;
}
inline float max_finite(const RVec<float>& values) {
  float result = missing();
  for (auto value : values)
    if (std::isfinite(value) && (!std::isfinite(result) || value > result)) result = value;
  return result;
}
inline RVec<int> count_erx(const RVec<float>& values) {
  RVec<int> result(values.size()/37, 0);
  for (size_t ch = 0; ch < values.size(); ++ch) if (std::isfinite(values[ch])) ++result[ch/37];
  return result;
}
inline RVec<int> count_above(const RVec<float>& values, double threshold) {
  RVec<int> result(values.size()/37, 0);
  for (size_t ch = 0; ch < values.size(); ++ch)
    if (std::isfinite(values[ch]) && values[ch] > threshold) ++result[ch/37];
  return result;
}
inline RVec<float> adc_window(RVec<float> values, bool has_low, double low, bool has_high, double high) {
  for (auto& value : values) {
    bool accepted = std::isfinite(value);
    if (has_low) accepted = accepted && value > low;
    if (has_high) accepted = accepted && (has_low ? value < high : value <= high);
    if (!accepted) value = missing();
  }
  return values;
}
}
#endif
