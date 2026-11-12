#ifndef HGCAL_CM_PLOT_HELPERS_HXX
#define HGCAL_CM_PLOT_HELPERS_HXX
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TTreeIndex.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace hgcal_cm_plot {
using ROOT::VecOps::RVec;
using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Vector=Eigen::VectorXd;

inline bool identity(unsigned run,int file,ULong64_t entry,ULong64_t id,ULong64_t event,
                     unsigned br,int bf,ULong64_t be,ULong64_t bi,ULong64_t bv,
                     unsigned expected_run,int expected_file,ULong64_t total) {
  if (run!=expected_run || file!=expected_file || entry>=total ||
      run!=br || file!=bf || entry!=be || id!=bi || event!=bv)
    throw std::runtime_error("Prepared/prediction physical event identity mismatch");
  return true;
}
inline void check_index(TTree* tree,ULong64_t expected) {
  auto index=dynamic_cast<TTreeIndex*>(tree->GetTreeIndex());
  if (!index || ULong64_t(index->GetN())!=expected)
    throw std::runtime_error("Invalid prepared source-entry index");
  const auto values=index->GetIndexValues();
  for (ULong64_t i=0;i<expected;++i) if (values[i]!=Long64_t(i))
    throw std::runtime_error("Missing or duplicate prepared source entry");
}
struct Pair {
  unsigned rows=0,cols=0;
  RVec<double> n,mx,my,m2;
  Pair()=default;
  Pair(unsigned r,unsigned c):rows(r),cols(c),n(r*c,0.),mx(r*c,0.),my(r*c,0.),m2(r*c,0.) {}
  void merge(const Pair& other) {
    for (unsigned j=0;j<n.size();++j) {
      if (!other.n[j]) continue;
      const double total=n[j]+other.n[j],dx=other.mx[j]-mx[j],dy=other.my[j]-my[j];
      m2[j]+=other.m2[j]+dx*dy*(n[j]*other.n[j]/total);
      mx[j]+=dx*(other.n[j]/total); my[j]+=dy*(other.n[j]/total); n[j]=total;
    }
  }
  void batch(const Matrix& x,const Matrix& y) {
    Matrix maskx=x.unaryExpr([](double v){return std::isfinite(v)?1.:0.;});
    Matrix masky=y.unaryExpr([](double v){return std::isfinite(v)?1.:0.;});
    Matrix zx=x.unaryExpr([](double v){return std::isfinite(v)?v:0.;});
    Matrix zy=y.unaryExpr([](double v){return std::isfinite(v)?v:0.;});
    Vector ox=zx.colwise().sum().transpose(),oy=zy.colwise().sum().transpose();
    Vector nx=maskx.colwise().sum().transpose(),ny=masky.colwise().sum().transpose();
    for (unsigned i=0;i<rows;++i) ox[i]=nx[i] ? ox[i]/nx[i] : 0.;
    for (unsigned j=0;j<cols;++j) oy[j]=ny[j] ? oy[j]/ny[j] : 0.;
    zx=(zx.rowwise()-ox.transpose()).cwiseProduct(maskx);
    zy=(zy.rowwise()-oy.transpose()).cwiseProduct(masky);
    Matrix counts=maskx.transpose()*masky;
    Matrix sx=zx.transpose()*masky,sy=maskx.transpose()*zy,xy=zx.transpose()*zy;
    for (unsigned i=0;i<rows;++i) for (unsigned j=0;j<cols;++j) {
      const auto k=i*cols+j; const double count=counts(i,j);
      if (!count) continue;
      n[k]=count; mx[k]=ox[i]+sx(i,j)/count; my[k]=oy[j]+sy(i,j)/count;
      m2[k]=xy(i,j)-sx(i,j)*sy(i,j)/count;
    }
  }
};
struct Moments {
  RVec<double> n,mean,m2;
  explicit Moments(unsigned p=0):n(p,0.),mean(p,0.),m2(p,0.) {}
  void add(unsigned c,double x) {
    const double delta=x-mean[c]; ++n[c]; mean[c]+=delta/n[c]; m2[c]+=delta*(x-mean[c]);
  }
  void merge(const Moments& other) {
    for (unsigned c=0;c<n.size();++c) if (other.n[c]) {
      const double total=n[c]+other.n[c],delta=other.mean[c]-mean[c];
      m2[c]+=other.m2[c]+delta*delta*(n[c]*other.n[c]/total);
      mean[c]+=delta*(other.n[c]/total); n[c]=total;
    }
  }
};
struct Histograms {
  Moments full,window;
  RVec<double> adc,mip,sum,flow;
  RVec<double> ranges;
  RVec<int> bins;
  Histograms()=default;
  Histograms(unsigned p,const RVec<double>& limits,const RVec<int>& counts):
    full(p),window(p),adc(counts[0],0.),mip(counts[1],0.),sum(counts[2],0.),flow(6,0.),ranges(limits),bins(counts) {}
  void fill(double value,RVec<double>& hist,unsigned kind) {
    const double lo=ranges[2*kind],hi=ranges[2*kind+1];
    if (value<lo) {++flow[2*kind];return;}
    if (value>=hi) {++flow[2*kind+1];return;}
    const unsigned b=std::min(unsigned((value-lo)/(hi-lo)*bins[kind]),unsigned(bins[kind]-1));
    ++hist[b];
  }
  void batch(const Matrix& x) {
    for (int i=0;i<x.rows();++i) {
      double total=0.;
      for (int c=0;c<x.cols();++c) {
        const double value=x(i,c); if (!std::isfinite(value)) continue;
        full.add(c,value); // independent of histogram limits: fixes legacy early-return bug
        if (value>=ranges[6] && value<=ranges[7]) window.add(c,value);
        fill(value,adc,0); fill(value,mip,1); total+=value;
      }
      fill(total,sum,2);
    }
  }
  void merge(const Histograms& other) {
    full.merge(other.full);window.merge(other.window);
    for (unsigned j=0;j<adc.size();++j) adc[j]+=other.adc[j];
    for (unsigned j=0;j<mip.size();++j) mip[j]+=other.mip[j];
    for (unsigned j=0;j<sum.size();++j) sum[j]+=other.sum[j];
    for (unsigned j=0;j<flow.size();++j) flow[j]+=other.flow[j];
  }
};
struct Row {RVec<float> raw,corrected,cm;ULong64_t entry;};
struct Buffer {
  Matrix raw,corrected,cm; unsigned size=0;
  Buffer(unsigned capacity,unsigned a,unsigned c,bool r,bool p,bool needcm):
    raw(capacity,r?a:0),corrected(capacity,p?a:0),cm(capacity,needcm?c:0) {}
  void add(const Row& row) {
    auto put=[&](Matrix& x,const RVec<float>& v){
      if (x.cols() && size_t(x.cols())!=v.size()) throw std::runtime_error("Plot vector width mismatch");
      for(int j=0;j<x.cols();++j) x(size,j)=v[j];
    };
    put(raw,row.raw);put(corrected,row.corrected);put(cm,row.cm);++size;
  }
};
class Collector {
  unsigned a,c,capacity;
  bool cr,cp,cc,hr,hp;
  RVec<double> ranges;RVec<int> bins;
  std::mutex mutex;
  void flush(Buffer& buffer) {
    if (!buffer.size) return;
    Matrix raw=buffer.raw.topRows(buffer.size),corr=buffer.corrected.topRows(buffer.size),cm=buffer.cm.topRows(buffer.size);
    Pair pr(cr?a:0,cr?a+c:0),pc(cp?a:0,cp?a+c:0),common(cc?c:0,cc?c:0);
    if (cr) {Matrix y(raw.rows(),a+c);y<<raw,cm;pr.batch(raw,y);}
    if (cp) {Matrix y(corr.rows(),a+c);y<<corr,cm;pc.batch(corr,y);}
    if (cc) common.batch(cm,cm);
    Histograms rh(hr?a:0,ranges,bins),ph(hp?a:0,ranges,bins);
    if (hr) rh.batch(raw);
    if (hp) ph.batch(corr);
    std::lock_guard<std::mutex> lock(mutex);
    if(cr) raw_cov.merge(pr);if(cp) corrected_cov.merge(pc);if(cc) cm_cov.merge(common);
    if(hr) raw_hist.merge(rh);if(hp) corrected_hist.merge(ph);
    buffer.size=0;
  }
public:
  Pair raw_cov,corrected_cov,cm_cov;
  Histograms raw_hist,corrected_hist;
  ULong64_t events=0;
  Collector(unsigned adc_size,unsigned cm_size,unsigned chunk,bool cov_raw,bool cov_pred,bool cov_cm,
            bool hist_raw,bool hist_pred,const RVec<double>& limits,const RVec<int>& counts):
    a(adc_size),c(cm_size),capacity(chunk),cr(cov_raw),cp(cov_pred),cc(cov_cm),hr(hist_raw),hp(hist_pred),
    ranges(limits),bins(counts),raw_cov(cr?a:0,cr?a+c:0),corrected_cov(cp?a:0,cp?a+c:0),cm_cov(cc?c:0,cc?c:0),
    raw_hist(hr?a:0,limits,counts),corrected_hist(hp?a:0,limits,counts) {}
  void consume(ROOT::RDF::RNode node) {
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<std::unordered_set<ULong64_t>> keys(node.GetNSlots());
    for(unsigned s=0;s<node.GetNSlots();++s) buffers.emplace_back(new Buffer(capacity,a,c,cr||hr,cp||hp,cr||cp||cc));
    node.ForeachSlot([&](unsigned slot,const Row& row){
      if (!keys[slot].insert(row.entry).second) throw std::runtime_error("Duplicate plotting source entry");
      auto& buffer=*buffers[slot];buffer.add(row);if(buffer.size==capacity) flush(buffer);
    },{"plot_payload"});
    std::unordered_set<ULong64_t> all;
    for(auto& slot:keys) for(auto entry:slot) if(!all.insert(entry).second)
      throw std::runtime_error("Duplicate plotting source entry across RDF slots");
    events=all.size();for(auto& buffer:buffers) flush(*buffer);
  }
};
}
#endif
