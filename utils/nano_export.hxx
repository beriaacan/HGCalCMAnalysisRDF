#ifndef HGCAL_CM_NANO_EXPORT_HXX
#define HGCAL_CM_NANO_EXPORT_HXX
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <TObjString.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>

namespace hgcal_cm_export {
using ROOT::VecOps::RVec;
template<class V> RVec<unsigned> copy_unsigned(const V& values) {
  return RVec<unsigned>(values.begin(), values.end());
}
struct Payload {
  ULong64_t entry, event;
  RVec<unsigned> fed, sequence, channel, adc, eligible;
};
inline std::unique_ptr<TFile> open(const std::string& path, const char* mode) {
  std::unique_ptr<TFile> f(TFile::Open(path.c_str(),mode));
  if (!f || f->IsZombie()) throw std::runtime_error("Cannot open Nano export file: "+path);
  return f;
}
inline TTree* get_tree(TFile* f, const char* name) {
  auto t=f->Get<TTree>(name);
  if (!t) throw std::runtime_error(std::string("Missing export tree: ")+name);
  return t;
}

class Prediction {
  using EventRows=std::unordered_map<ULong64_t,Long64_t>;
  static EventRows event_index(TTree* tree,ULong64_t count,ULong64_t source_entries,const std::string& path) {
    if (ULong64_t(tree->GetEntries())!=count)
      throw std::runtime_error("Prediction event count changed: "+path);
    tree->SetCacheSize(0);
    EventRows rows; rows.reserve(count);
    std::vector<ULong64_t> cache_entries;cache_entries.reserve(count);
    // Read exact unsigned integers, rather than a floating-point formula or
    // rdfentry_. Prepared IDs are internal keys under ROOT implicit MT.
    TTreeReader scan(tree);
    TTreeReaderValue<ULong64_t> event(scan,"event"),entry(scan,"source_entry");
    while(scan.Next()) {
      if (*entry>=source_entries || !rows.emplace(*event,scan.GetCurrentEntry()).second)
        throw std::runtime_error("Duplicate Nano event or out-of-range cache ID in predictions: "+path);
      cache_entries.push_back(*entry);
    }
    if (cache_entries.size()!=count) throw std::runtime_error("Could not read all prediction identity rows: "+path);
    std::sort(cache_entries.begin(),cache_entries.end());
    if (std::adjacent_find(cache_entries.begin(),cache_entries.end())!=cache_entries.end())
      throw std::runtime_error("Duplicate prediction cache source entry: "+path);
    return rows;
  }
  std::unique_ptr<TFile> file;
  TTree* tree;
  EventRows event_rows;
  TTreeReader reader;
  TTreeReaderValue<UInt_t> run;
  TTreeReaderValue<Int_t> source;
  TTreeReaderValue<ULong64_t> entry, id, event;
  TTreeReaderArray<float> prediction, residual;
public:
  std::string method, module;
  unsigned fed, sequence, channels;
  ULong64_t seen=0, expected;
  Prediction(const std::string& m,const std::string& name,unsigned f,unsigned s,unsigned nch,
             const std::string& path,ULong64_t count,ULong64_t original_entries)
    :file(open(path,"READ")), tree(get_tree(file.get(),"Predictions")),
     event_rows(event_index(tree,count,original_entries,path)), reader(tree),
     run(reader,"source_run"),source(reader,"source_file_index"),entry(reader,"source_entry"),
     id(reader,"event_id"),event(reader,"event"),prediction(reader,"prediction"),residual(reader,"adc_corrected"),
     method(m),module(name),fed(f),sequence(s),channels(nch),expected(count) {
    // Keep one basket per branch plus the integer index, not one large ROOT
    // read-ahead cache for every module/method in the Nano.
  }
  bool load(ULong64_t physical_entry,ULong64_t original_event,unsigned expected_run,int expected_file,
            ULong64_t offset,std::vector<float>& values) {
    auto matched=event_rows.find(original_event);
    if (matched==event_rows.end()) return false;
    auto row=matched->second;
    if (reader.SetEntry(row)!=TTreeReader::kEntryValid)
      throw std::runtime_error("Cannot read indexed prediction row: "+module+"/"+method);
    if (*run!=expected_run || *source!=expected_file || *event!=original_event || *id!=offset+*entry)
      throw std::runtime_error("Nano/prediction identity mismatch: "+module+"/"+method+
        "; Nano entry="+std::to_string(physical_entry)+", event="+std::to_string(original_event)+
        "; expected run/file="+std::to_string(expected_run)+"/"+std::to_string(expected_file)+
        "; stored run/file/cache-entry/event-id/event="+std::to_string(*run)+"/"+std::to_string(*source)+"/"+
        std::to_string(*entry)+"/"+std::to_string(*id)+"/"+std::to_string(*event));
    if (prediction.GetSize()!=channels || residual.GetSize()!=channels)
      throw std::runtime_error("Prediction channel width changed: "+module+"/"+method);
    ++seen;
    values.assign(channels,std::numeric_limits<float>::quiet_NaN());
    for (unsigned c=0;c<channels;++c) {
      bool p=std::isfinite(prediction[c]),r=std::isfinite(residual[c]);
      if (p!=r || (!p && (!std::isnan(prediction[c]) || !std::isnan(residual[c]))))
        throw std::runtime_error("Invalid prediction/residual validity: "+module+"/"+method);
      if (p) values[c]=prediction[c];
    }
    return true;
  }
};

struct Output {
  std::string method;
  std::vector<float> correction, corrected;
  std::vector<unsigned char> valid;
  TBranch *bcorrection=nullptr,*bcorrected=nullptr,*bvalid=nullptr;
  ULong64_t count=0;
  Output(TTree* tree,const std::string& name):method(name) {
    std::string a="HGCDigi_adcCorrection_"+method,b="HGCDigi_adcCorrected_"+method,
                c="HGCDigi_adcCorrectionValid_"+method;
    if (tree->GetBranch(a.c_str()) || tree->GetBranch(b.c_str()) || tree->GetBranch(c.c_str()))
      throw std::runtime_error("Nano already contains correction branches for "+method);
    bcorrection=tree->Branch(a.c_str(),&correction);
    bcorrected=tree->Branch(b.c_str(),&corrected);
    bvalid=tree->Branch(c.c_str(),&valid);
    if (!bcorrection || !bcorrected || !bvalid)
      throw std::runtime_error("Could not create Nano correction branches");
  }
  void reset(size_t n) {
    correction.assign(n,std::numeric_limits<float>::quiet_NaN());
    corrected=correction;valid.assign(n,0);
  }
  void fill() {
    if (bcorrection->Fill()<0 || bcorrected->Fill()<0 || bvalid->Fill()<0)
      throw std::runtime_error("Failed writing Nano correction branch");
  }
};

class Writer {
  std::unique_ptr<TFile> file;
  TTree* tree;
  ULong64_t expected,offset;
  unsigned run;
  int source;
  bool finished=false;
  unsigned progress_events;
  std::unordered_set<ULong64_t> nano_events;
  std::vector<std::unique_ptr<Output>> outputs;
  std::vector<std::unique_ptr<Prediction>> inputs;
  std::unordered_set<std::string> input_keys;
  Output& output(const std::string& method) {
    for (auto& p:outputs) if (p->method==method) return *p;
    throw std::runtime_error("Unknown export method: "+method);
  }
  void append(const Payload& row) {
    if (row.entry!=events || events>=expected)
      throw std::runtime_error("Nano export event order/count changed; export must run sequentially");
    if (!nano_events.insert(row.event).second)
      throw std::runtime_error("Duplicate event number in original Nano; cannot export an ambiguous event join");
    const auto n=row.adc.size();
    if (row.fed.size()!=n || row.sequence.size()!=n || row.channel.size()!=n || row.eligible.size()!=n)
      throw std::runtime_error("Nano digi-vector lengths differ");
    for (auto& p:outputs) p->reset(n);
    // One physical module lookup per event. Digi order may vary and may contain
    // unrequested modules; never scatter using a positional channel assumption.
    std::unordered_map<ULong64_t,std::vector<size_t>> digis;
    bool mapped=false;
    std::vector<float> values;
    for (auto& input:inputs) {
      bool found=input->load(row.entry,row.event,run,source,offset,values);
      if (!found) continue;
      if (!mapped) {
        for (size_t i=0;i<n;++i) digis[(ULong64_t(row.fed[i])<<32)|row.sequence[i]].push_back(i);
        mapped=true;
      }
      auto it=digis.find((ULong64_t(input->fed)<<32)|input->sequence);
      std::vector<bool> seen(input->channels,false);
      auto& out=output(input->method);
      if (it!=digis.end()) for (size_t i:it->second) {
        auto c=row.channel[i];
        if (c>=input->channels || seen[c])
          throw std::runtime_error("Duplicate/out-of-range Nano module channel: "+input->module);
        seen[c]=true;
        if (!found || !std::isfinite(values[c])) continue;
        if (!row.eligible[i])
          throw std::runtime_error("Saved correction refers to a non-ADC Nano digi: "+input->module);
        float corrected=float(row.adc[i])-values[c];
        if (!std::isfinite(corrected)) throw std::runtime_error("Nonfinite exported corrected ADC");
        out.correction[i]=values[c];out.corrected[i]=corrected;out.valid[i]=1;++out.count;
      }
      if (found) for (unsigned c=0;c<input->channels;++c)
        if (std::isfinite(values[c]) && !seen[c])
          throw std::runtime_error("Saved correction channel missing from Nano: "+input->module);
    }
    for (auto& p:outputs) p->fill();
    ++events;
    if (progress_events && (events%progress_events==0 || events==expected))
      std::cout<<"  Appended correction branches: "<<events<<" / "<<expected<<" original events"<<std::endl;
  }
public:
  ULong64_t events=0;
  Writer(const std::string& path,const std::string& uuid,ULong64_t entries,unsigned source_run,
         int source_file,ULong64_t entry_offset,const std::vector<std::string>& methods,unsigned progress_step)
    :file(open(path,"UPDATE")),tree(get_tree(file.get(),"Events")),expected(entries),offset(entry_offset),
     run(source_run),source(source_file),progress_events(progress_step) {
    if (ULong64_t(tree->GetEntries())!=entries || std::string(file->GetUUID().AsString())!=uuid)
      throw std::runtime_error("Copied Nano UUID/event count differs from original source");
    if (file->Get("HGCalCMAnalysisExport"))
      throw std::runtime_error("Source Nano already contains HGCal CM export metadata");
    if (methods.empty()) throw std::runtime_error("No export methods requested");
    std::unordered_set<std::string> unique;
    for (const auto& method:methods) {
      if (!unique.insert(method).second) throw std::runtime_error("Duplicate export method");
      outputs.emplace_back(new Output(tree,method));
    }
  }
  void add(const std::string& method,const std::string& module,unsigned fed,unsigned sequence,unsigned nch,
           const std::string& path,ULong64_t selected_events) {
    if (events || finished) throw std::runtime_error("Cannot add predictions after Nano export has started");
    output(method);
    auto key=method+":"+std::to_string(fed)+":"+std::to_string(sequence);
    if (!input_keys.insert(key).second) throw std::runtime_error("Duplicate export module/method identity");
    inputs.emplace_back(new Prediction(method,module,fed,sequence,nch,path,selected_events,expected));
  }
  void consume(ROOT::RDF::RNode node) {
    if (ROOT::IsImplicitMTEnabled()) throw std::runtime_error("Nano export requires sequential RDF traversal");
    node.Foreach([this](const Payload& row){append(row);},{"hgcal_export_payload"});
  }
  ULong64_t valid(const std::string& method) { return output(method).count; }
  void finish(const std::string& provenance) {
    if (finished || events!=expected) throw std::runtime_error("Incomplete Nano export event count");
    for (const auto& p:inputs) if (p->seen!=p->expected)
      throw std::runtime_error("Not every saved prediction row was matched to its Nano event: "+p->module+"/"+p->method);
    for (const auto& p:outputs)
      for (auto b:{p->bcorrection,p->bcorrected,p->bvalid})
        if (ULong64_t(b->GetEntries())!=expected) throw std::runtime_error("Incomplete Nano export branch");
    file->cd();
    if (tree->Write("",TObject::kOverwrite)<=0) throw std::runtime_error("Could not write augmented Events header");
    TObjString metadata(provenance.c_str());
    if (metadata.Write("HGCalCMAnalysisExport")<=0) throw std::runtime_error("Could not write Nano export provenance");
    file->Close();
    if (file->TestBit(TFile::kWriteError)) throw std::runtime_error("Nano export write error");
    finished=true;
  }
};
}
#endif
