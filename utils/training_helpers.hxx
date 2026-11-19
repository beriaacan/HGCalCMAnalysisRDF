#ifndef HGCAL_CM_TRAINING_HELPERS_HXX
#define HGCAL_CM_TRAINING_HELPERS_HXX
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace hgcal_cm_training {
using ROOT::VecOps::RVec;
inline std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
inline double uniform(std::uint64_t entry, std::uint64_t salt, std::uint64_t seed) {
  return double(mix(entry ^ salt ^ seed) >> 11) * 0x1.0p-53;
}
inline unsigned split(std::uint64_t entry, std::uint64_t salt, std::uint64_t seed, double validation) {
  return uniform(entry, salt, seed) < validation ? 1u : 0u;
}
inline bool retained(std::uint64_t entry, std::uint64_t salt, std::uint64_t seed,
                     unsigned partition, double train_fraction, double validation_fraction) {
  return uniform(entry, salt ^ 0xd1b54a32d192ed03ULL, seed)
         < (partition == 0 ? train_fraction : validation_fraction);
}
inline RVec<float> padded(const RVec<float>& values, unsigned width, float fill) {
  if (values.size() > width) throw std::runtime_error("Feature exceeds configured padded width");
  RVec<float> out(width, fill);
  for (size_t i=0; i<values.size(); ++i) if (std::isfinite(values[i])) out[i] = values[i];
  return out;
}
// Zero suppression of the ADC inputs: 1 where a finite value <= threshold is suppressed.
// A random fraction of channels per event is exempt; the draw is a hash of (module salt,
// event, channel), identical to the old HGCalCMAnalysis random_unsuppressed_mask.
inline double zs_uniform(std::uint64_t event, std::uint64_t channel, std::uint64_t salt) {
  std::uint64_t x = event*0x9E3779B97F4A7C15ULL + channel*0xBF58476D1CE4E5B9ULL + salt;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27; x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  return double(x >> 11) / double(1ULL << 53);
}
inline RVec<float> zs_mask(const RVec<float>& adc, double threshold, double fraction,
                           std::uint64_t event, std::uint64_t salt) {
  RVec<float> mask(adc.size(), 0.f);
  for (size_t c=0; c<adc.size(); ++c)
    if (std::isfinite(adc[c]) && adc[c] <= threshold && !(fraction > 0 && zs_uniform(event, c, salt) < fraction))
      mask[c] = 1.f;
  return mask;
}
inline RVec<float> zero_suppressed(const RVec<float>& adc, const RVec<float>& mask) {
  RVec<float> out(adc);
  for (size_t c=0; c<out.size(); ++c) if (mask[c] > 0.f) out[c] = 0.f;
  return out;
}
inline void append(RVec<float>& out, const RVec<float>& values) { out.insert(out.end(), values.begin(), values.end()); }
inline unsigned count_targets(const RVec<float>& values) {
  unsigned count=0;
  for (float value : values) if (std::isfinite(value)) ++count;
  return count;
}
struct Moment {
  std::uint64_t count = 0;
  double mean = 0., m2 = 0.;
  void add(double x, std::uint64_t weight=1) {
    if (!weight) return;
    if (!std::isfinite(x)) throw std::runtime_error("Nonfinite training statistic");
    const auto total = count + weight;
    const double delta = x-mean;
    mean += delta * (double(weight)/double(total));
    m2 += delta*delta * (double(count)*double(weight)/double(total));
    count = total;
  }
  void merge(const Moment& other) {
    if (!other.count) return;
    if (!count) { *this = other; return; }
    const auto total = count + other.count;
    const double delta = other.mean - mean;
    m2 += other.m2 + delta*delta*(double(count)*double(other.count)/double(total));
    mean += delta*(double(other.count)/double(total));
    count = total;
  }
};
struct Row {
  RVec<float> global, targets;
  unsigned partition;
  bool keep;
};
struct Statistics {
  std::vector<Moment> inputs, targets;
  std::vector<std::uint64_t> events, kept_events, samples, channel_counts;
  RVec<int> adc_columns, local_kinds, adc_target;
  RVec<float> areas;
  unsigned global_size = 0, width = 0;
  float fill = 0.f;
  bool fit = true;
  Statistics() = default;
  Statistics(unsigned nfeatures, unsigned max_channels, RVec<int> adc, RVec<int> own_mask,
             RVec<int> local, RVec<float> cell_areas, float missing, bool fit_inputs)
    : inputs(nfeatures), targets(max_channels), events(2,0), kept_events(2,0), samples(2,0),
      channel_counts(2*max_channels,0), adc_columns(std::move(adc)), local_kinds(std::move(local)),
      areas(std::move(cell_areas)), global_size(nfeatures-local_kinds.size()), width(max_channels),
      fill(missing), fit(fit_inputs) {
    adc_target = RVec<int>(global_size, -1);
    for (size_t c=0; c<adc_columns.size(); ++c)
      if (adc_columns[c]>=0) adc_target.at(adc_columns[c]) = int(c);
    // A target's own ZS mask bit is hidden like its own ADC.
    for (size_t c=0; c<own_mask.size(); ++c)
      if (own_mask[c]>=0) adc_target.at(own_mask[c]) = int(c);
  }
  void add(const Row& row) {
    if (row.partition>1 || row.targets.size()!=areas.size() || row.targets.size()>width) throw std::runtime_error("Invalid training row");
    const auto s = row.partition;
    ++events[s];
    if (!row.keep) return;
    ++kept_events[s];
    std::uint64_t n = 0;
    for (size_t c=0; c<row.targets.size(); ++c) if (std::isfinite(row.targets[c])) {
      ++n; ++channel_counts[s*width+c];
      if (s==0) targets[c].add(row.targets[c]);
    }
    samples[s] += n;
    if (s!=0 || !fit || !n) return;
    if (row.global.size()!=global_size) throw std::runtime_error("Global feature width mismatch");
    for (size_t j=0; j<global_size; ++j) {
      const int c = adc_target[j];
      const bool hidden = c>=0 && size_t(c)<row.targets.size() && std::isfinite(row.targets[c]);
      inputs[j].add(row.global[j], n-std::uint64_t(hidden));
      if (hidden) inputs[j].add(fill);
    }
    for (size_t c=0; c<row.targets.size(); ++c) if (std::isfinite(row.targets[c])) {
      for (size_t k=0; k<local_kinds.size(); ++k) {
        double value = local_kinds[k]==0 ? double(c%37) : local_kinds[k]==1 ? double(c/37) : double(areas.at(c));
        inputs[global_size+k].add(value);
      }
    }
  }
  void merge(const Statistics& other) {
    if (inputs.size()!=other.inputs.size() || width!=other.width) throw std::runtime_error("Statistic shapes disagree");
    for (size_t i=0; i<inputs.size(); ++i) inputs[i].merge(other.inputs[i]);
    for (size_t i=0; i<targets.size(); ++i) targets[i].merge(other.targets[i]);
    for (size_t i=0; i<2; ++i) { events[i]+=other.events[i]; kept_events[i]+=other.kept_events[i]; samples[i]+=other.samples[i]; }
    for (size_t i=0; i<channel_counts.size(); ++i) channel_counts[i]+=other.channel_counts[i];
  }
};
inline ROOT::RDF::RResultPtr<Statistics> book_statistics(
    ROOT::RDF::RNode node, unsigned nfeatures, unsigned width,
    const RVec<int>& adc_columns, const RVec<int>& own_mask, const RVec<int>& local_kinds,
    const RVec<float>& areas, float fill, bool fit) {
  auto add = [](Statistics& stats, const Row& row) { stats.add(row); };
  auto merge = [](std::vector<Statistics>& slots) { for (size_t i=1; i<slots.size(); ++i) slots[0].merge(slots[i]); };
  return node.Aggregate(add, merge, "training_payload",
                        Statistics(nfeatures,width,adc_columns,own_mask,local_kinds,areas,fill,fit));
}
}
#endif
