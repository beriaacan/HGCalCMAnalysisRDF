#ifndef HGCAL_CM_INFERENCE_IO_HXX
#define HGCAL_CM_INFERENCE_IO_HXX
#include <TFile.h>
#include <TTree.h>
#include <ROOT/RVec.hxx>
#include <memory>
#include <vector>
#include <stdexcept>

namespace hgcal_cm_inference {
class Writer {
  std::unique_ptr<TFile> file;
  TTree* tree;
  UInt_t run=0;
  Int_t source=0;
  ULong64_t entry=0, id=0, event=0;
  std::vector<float> prediction, corrected;
  unsigned channels;
public:
  Writer(const std::string& path, unsigned nch, unsigned flush): channels(nch) {
    file.reset(TFile::Open(path.c_str(), "RECREATE"));
    if (!file || file->IsZombie()) throw std::runtime_error("Cannot create prediction ROOT file");
    file->cd();
    tree=new TTree("Predictions", "DNN correction in physical ADC units");
    tree->SetAutoFlush(flush);
    tree->Branch("source_run", &run);
    tree->Branch("source_file_index", &source);
    tree->Branch("source_entry", &entry);
    tree->Branch("event_id", &id);
    tree->Branch("event", &event);
    tree->Branch("prediction", &prediction);
    tree->Branch("adc_corrected", &corrected);
  }
  void append(const ROOT::VecOps::RVec<ULong64_t>& ids,
              const ROOT::VecOps::RVec<float>& predictions,
              const ROOT::VecOps::RVec<float>& corrections) {
    if (ids.size()%5 || predictions.size()!=ids.size()/5*channels || corrections.size()!=predictions.size())
      throw std::runtime_error("Inconsistent inference output shapes");
    for (size_t r=0; r<ids.size()/5; ++r) {
      run=ids[5*r]; source=ids[5*r+1]; entry=ids[5*r+2]; id=ids[5*r+3]; event=ids[5*r+4];
      prediction.assign(predictions.begin()+r*channels, predictions.begin()+(r+1)*channels);
      corrected.assign(corrections.begin()+r*channels, corrections.begin()+(r+1)*channels);
      if (tree->Fill()<0) throw std::runtime_error("Failed writing prediction row");
    }
  }
  void finish() {
    file->cd();
    if (tree->Write("", TObject::kOverwrite)<=0) throw std::runtime_error("Failed writing prediction tree");
    file->Close();
    if (file->TestBit(TFile::kWriteError)) throw std::runtime_error("Prediction ROOT file write error");
  }
};
}
#endif
