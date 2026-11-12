#ifndef HGCAL_CM_ANALYTIC_HELPERS_HXX
#define HGCAL_CM_ANALYTIC_HELPERS_HXX
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

// RDF owns event scheduling; Eigen products are deliberately single-threaded.
// There is one module's covariance storage, rather than one copy per RDF slot.
namespace hgcal_cm_analytic {
using ROOT::VecOps::RVec;
using Matrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

struct Moment {
  std::uint64_t n = 0;
  Vector mean, cross, minimum, maximum;
  Matrix scatter;
  double target_mean = 0.;
  explicit Moment(unsigned p) : mean(Vector::Zero(p)), cross(Vector::Zero(p)),
    minimum(Vector::Constant(p,INFINITY)), maximum(Vector::Constant(p,-INFINITY)),
    scatter(Matrix::Zero(p,p)) {}
  void merge(std::uint64_t count, const Vector& mx, double my, const Matrix& xx,
             const Vector& xy, const Vector& low, const Vector& high) {
    if (!count) return;
    if (!n) {
      n=count; mean=mx; target_mean=my; scatter=xx; cross=xy; minimum=low; maximum=high;
      return;
    }
    const double total=double(n)+double(count), factor=double(n)*double(count)/total;
    const Vector delta=mx-mean;
    scatter += xx;
    scatter.noalias() += factor*delta*delta.transpose();
    cross += xy + factor*(my-target_mean)*delta;
    mean += (double(count)/total)*delta;
    target_mean += (double(count)/total)*(my-target_mean);
    minimum=minimum.cwiseMin(low); maximum=maximum.cwiseMax(high);
    n+=count;
  }
};

struct Buffer {
  Matrix x, y;
  unsigned size=0;
  Buffer(unsigned capacity, unsigned p, unsigned nch) : x(capacity,p), y(capacity,nch) {}
  void add(const RVec<float>& global, const RVec<float>& targets, const RVec<int>& columns) {
    if (targets.size()!=size_t(y.cols())) throw std::runtime_error("Analytic target width changed");
    for (unsigned j=0;j<columns.size();++j) {
      const double value=global.at(columns[j]);
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite analytic predictor");
      x(size,j)=value;
    }
    for (unsigned c=0;c<targets.size();++c) y(size,c)=targets[c];
    ++size;
  }
};

class Fitter {
  RVec<int> columns;
  unsigned nch, capacity;
  std::vector<Moment> moments;
  std::mutex mutex;

  void flush(Buffer& buffer) {
    const unsigned n=buffer.size, p=columns.size();
    if (!n) return;
    const auto x=buffer.x.topRows(n);
    const Vector origin=x.colwise().mean();
    Matrix z=x.rowwise()-origin.transpose();
    const Vector total_z=z.colwise().sum();
    const Vector all_low=x.colwise().minCoeff(), all_high=x.colwise().maxCoeff();
    Matrix common=z.transpose()*z;
    for (unsigned c=0;c<nch;++c) {
      unsigned count=0;
      double sum_y=0.;
      for (unsigned i=0;i<n;++i) if (std::isfinite(buffer.y(i,c))) {
        ++count; sum_y+=buffer.y(i,c);
      }
      if (!count) continue;
      const double my=sum_y/count;
      Vector low=all_low, high=all_high, sum=total_z;
      Matrix xx=common;
      if (count!=n) {
        const bool use_accepted=count<=n/2;
        Matrix small(use_accepted ? count : n-count,p);
        unsigned at=0;
        low=Vector::Constant(p,INFINITY); high=Vector::Constant(p,-INFINITY);
        for (unsigned i=0;i<n;++i) {
          const bool valid=std::isfinite(buffer.y(i,c));
          if (valid) {
            low=low.cwiseMin(x.row(i).transpose());
            high=high.cwiseMax(x.row(i).transpose());
          }
          if (valid==use_accepted) small.row(at++)=z.row(i);
        }
        if (use_accepted) {
          sum=small.colwise().sum();
          xx.noalias()=small.transpose()*small;
        } else {
          sum-=small.colwise().sum().transpose();
          xx.noalias()-=small.transpose()*small;
        }
      }
      // Both moments below describe precisely the accepted target population.
      xx.noalias()-=(sum*sum.transpose())/double(count);
      Vector centered_y=Vector::Zero(n);
      for (unsigned i=0;i<n;++i) if (std::isfinite(buffer.y(i,c))) centered_y[i]=buffer.y(i,c)-my;
      const Vector xy=z.transpose()*centered_y - sum*(centered_y.sum()/double(count));
      const Vector mx=origin+sum/double(count);
      std::lock_guard<std::mutex> lock(mutex);
      moments[c].merge(count,mx,my,xx,xy,low,high);
    }
    buffer.size=0;
  }

public:
  Fitter(const RVec<int>& selected_columns, unsigned channels, unsigned chunk)
    : columns(selected_columns), nch(channels), capacity(chunk) {
    if (!nch || !capacity) throw std::runtime_error("Invalid analytic dimensions");
    moments.reserve(nch);
    for (unsigned c=0;c<nch;++c) moments.emplace_back(columns.size());
  }
  void consume(ROOT::RDF::RNode node) {
    std::vector<std::unique_ptr<Buffer>> buffers;
    for (unsigned s=0;s<node.GetNSlots();++s) buffers.emplace_back(new Buffer(capacity,columns.size(),nch));
    node.ForeachSlot([&](unsigned slot,const RVec<float>& global,const RVec<float>& targets) {
      auto& buffer=*buffers.at(slot);
      buffer.add(global,targets,columns);
      if (buffer.size==capacity) flush(buffer);
    }, {"training_global_features","adc_selected"});
    for (auto& buffer:buffers) flush(*buffer);
  }
  std::uint64_t count(unsigned c) const { return moments.at(c).n; }
  double target_mean(unsigned c) const { return moments.at(c).target_mean; }
  RVec<double> mean(unsigned c) const {
    const auto& value=moments.at(c).mean; return {value.data(),value.data()+value.size()};
  }
  RVec<double> cross(unsigned c) const {
    const auto& value=moments.at(c).cross; return {value.data(),value.data()+value.size()};
  }
  RVec<double> scatter(unsigned c) const {
    const auto& value=moments.at(c).scatter; return {value.data(),value.data()+value.size()};
  }
  RVec<int> varying(unsigned c) const {
    const auto& value=moments.at(c); RVec<int> result;
    for (unsigned j=0;j<columns.size();++j) if (value.minimum[j]<value.maximum[j]) result.push_back(j);
    return result;
  }
};

struct ResidualMoment {
  std::uint64_t eligible=0, n=0;
  double target_mean=0., target_m2=0., residual_mean=0., residual_m2=0.;
  void add(double target,double prediction,bool fitted) {
    ++eligible;
    if (!fitted) return;
    if (!std::isfinite(prediction)) throw std::runtime_error("Nonfinite analytic validation prediction");
    ++n;
    const double dy=target-target_mean, dr=(target-prediction)-residual_mean;
    target_mean+=dy/double(n); target_m2+=dy*(target-target_mean);
    residual_mean+=dr/double(n); residual_m2+=dr*((target-prediction)-residual_mean);
  }
  void merge(const ResidualMoment& other) {
    eligible+=other.eligible;
    if (!other.n) return;
    const double total=double(n)+double(other.n), factor=double(n)*double(other.n)/total;
    target_m2+=other.target_m2+factor*std::pow(other.target_mean-target_mean,2);
    residual_m2+=other.residual_m2+factor*std::pow(other.residual_mean-residual_mean,2);
    target_mean+=(double(other.n)/total)*(other.target_mean-target_mean);
    residual_mean+=(double(other.n)/total)*(other.residual_mean-residual_mean);
    n+=other.n;
  }
};

class Evaluator {
  RVec<int> columns, fitted;
  Matrix weights;
  Vector intercept;
  unsigned capacity, nch;
  std::mutex mutex;
  void flush(Buffer& buffer) {
    if (!buffer.size) return;
    Matrix predictions=buffer.x.topRows(buffer.size)*weights.transpose();
    predictions.rowwise()+=intercept.transpose();
    std::vector<ResidualMoment> local(nch);
    for (unsigned i=0;i<buffer.size;++i) for (unsigned c=0;c<nch;++c)
      if (std::isfinite(buffer.y(i,c))) local[c].add(buffer.y(i,c),predictions(i,c),fitted[c]);
    std::lock_guard<std::mutex> lock(mutex);
    for (unsigned c=0;c<nch;++c) moments[c].merge(local[c]);
    buffer.size=0;
  }
public:
  std::vector<ResidualMoment> moments;
  Evaluator(const RVec<int>& selected_columns,const RVec<double>& coefficients,
            const RVec<double>& offsets,const RVec<int>& available,unsigned chunk)
    : columns(selected_columns), fitted(available), weights(offsets.size(),columns.size()),
      intercept(offsets.size()), capacity(chunk), nch(offsets.size()), moments(nch) {
    if (!capacity || fitted.size()!=nch || coefficients.size()!=nch*columns.size())
      throw std::runtime_error("Invalid analytic evaluator dimensions");
    for (unsigned c=0;c<nch;++c) {
      intercept[c]=fitted[c] ? offsets[c] : 0.;
      for (unsigned j=0;j<columns.size();++j) weights(c,j)=fitted[c] ? coefficients[c*columns.size()+j] : 0.;
    }
  }
  void consume(ROOT::RDF::RNode node) {
    std::vector<std::unique_ptr<Buffer>> buffers;
    for (unsigned s=0;s<node.GetNSlots();++s) buffers.emplace_back(new Buffer(capacity,columns.size(),nch));
    node.ForeachSlot([&](unsigned slot,const RVec<float>& global,const RVec<float>& targets) {
      auto& buffer=*buffers.at(slot); buffer.add(global,targets,columns);
      if (buffer.size==capacity) flush(buffer);
    }, {"training_global_features","adc_selected"});
    for (auto& buffer:buffers) flush(*buffer);
  }
};
}
#endif
