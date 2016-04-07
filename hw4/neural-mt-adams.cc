#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "ntag/word-embed.h"
#include "cnn/cfsm-builder.h"
#include "cnn/hsm-builder.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>

#include <boost/program_options.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace cnn;

float DROPOUT = 0.0;
unsigned IN_LAYERS = 1;
unsigned OUT_LAYERS = 2;
unsigned CHAR_DIM = 32;
unsigned REMINDER_DIM = 64;
unsigned LSTM_DIM = 128;
unsigned VOCAB_SIZE = 0;

cnn::Dict xd;
cnn::Dict yd;
int kSOS;
int kSOS2;
int kEOS;
int kEOS2;

FactoredSoftmaxBuilder* cfsm = nullptr;

namespace po = boost::program_options;
void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("train_1", po::value<string>(), "Training corpus 1")
        ("train_2", po::value<string>(), "Training corpus 2")
        ("dev_1", po::value<string>(), "Dev corpus 1")
        ("dev_2", po::value<string>(), "Dev corpus 2")
        ("test_1", po::value<string>(), "Test corpus 1")
        ("test_2", po::value<string>(), "Test corpus 2")
        ("model,m", po::value<string>(), "Load model params")
        ("dropout,D", po::value<float>(), "Dropout rate")
        ("clusters,c", po::value<string>(), "word cluster file for class factored softmax")
        ("in_layers,i", po::value<unsigned>()->default_value(1), "Number of layers in input BiLSTM")
        ("out_layers,o", po::value<unsigned>()->default_value(2), "Number of layers in output LSTM")
        ("char_dim", po::value<unsigned>()->default_value(32), "Word dimensions")
        ("lstm_dim", po::value<unsigned>()->default_value(128), "Tag dimensions")
        ("learn,x", "Learn parameters")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("train_1") == 0 || conf->count("train_2") == 0) {
    cerr << "Please specify --train_1 and --train_2 : this is required to determine the vocabulary mapping, even at test time\n";
    exit(1);
  }
}
template <class Builder>
struct AEModel {
  WordEmbed* we_x;
  WordEmbed* we_y;
  Builder l2rbuilder;
  Builder r2lbuilder;
  Builder outbuilder;
  Parameters* p_enc2dec;
  Parameters* p_decbias;
  Parameters* p_dec2reminder;
  Parameters* p_enc2out;
  Parameters* p_outbias;
  explicit AEModel(Model& model, WordEmbed* pwe_x, WordEmbed* pwe_y) :
      we_x(pwe_x),
      we_y(pwe_y),
      l2rbuilder(IN_LAYERS, CHAR_DIM, LSTM_DIM, &model),
      r2lbuilder(IN_LAYERS, CHAR_DIM, LSTM_DIM, &model),
      outbuilder(OUT_LAYERS, CHAR_DIM, LSTM_DIM, &model) {
//      outbuilder(OUT_LAYERS, CHAR_DIM, LSTM_DIM + REMINDER_DIM, &model) {
    p_enc2dec = model.add_parameters({LSTM_DIM * OUT_LAYERS, 2 * LSTM_DIM});
    p_decbias = model.add_parameters({LSTM_DIM * OUT_LAYERS});
    p_enc2out = model.add_parameters({VOCAB_SIZE, LSTM_DIM});
    p_outbias = model.add_parameters({VOCAB_SIZE});
    //p_dec2reminder = model.add_parameters({REMINDER_DIM, 2 * LSTM_DIM});
  }

  Expression EmbedBiLSTM(ComputationGraph& cg, const vector<int>& toks, bool apply_dropout) {
    const unsigned slen = toks.size();
    we_x->new_graph(cg);
    l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);  // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    if (apply_dropout) {
      l2rbuilder.set_dropout(DROPOUT);
      r2lbuilder.set_dropout(DROPOUT);
    } else {
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
    }

    // read sequence from left to right
    vector<Expression> left;
    l2rbuilder.add_input(we_x->embed(kSOS));
    for (unsigned t = 0; t < slen; ++t)
      l2rbuilder.add_input(we_x->embed(toks[t]));
      left.push_back(l2rbuilder.back());
    //Expression fr = l2rbuilder.back(); // dropout not applied

    // read sequence from right to left
    deque<Expression> right;
    r2lbuilder.add_input(we_x->embed(kEOS));
    for (unsigned t = 0; t < slen; ++t)
      r2lbuilder.add_input(we_x->embed(toks[slen - t - 1]));
      right.push_front(r2lbuilder.back());
    //Expression fl = r2lbuilder.back();
    assert (left.size() == right.size());
    Expression fr = sum(left) / ((float) left.size());
    Expression fl = sum(right) / ((float) right.size());

    return concatenate({fr, fl});
  }

  // return tags
  void BuildGraph(const vector<int>& toks_source, const vector<int>& toks_target, ComputationGraph& cg, bool apply_dropout) {
    assert (cfsm);
    const unsigned slen = toks_target.size();
    // enc {2 * LSTM_DIM}
    //cerr << "Milestone 3" << endl;
    Expression enc = EmbedBiLSTM(cg, toks_source, apply_dropout);
    //cerr << "Milestone 4" << endl;
    // enc2dec = {LSTM_DIM * OUT_LAYERS, 2 * LSTM_DIM}
    // V = {REMINDER_DIM, 2 * LSTM_DIM}
    Expression enc2dec = parameter(cg, p_enc2dec);
    Expression decbias = parameter(cg, p_decbias);
    Expression c0 = affine_transform({decbias, enc2dec, enc});
    vector<Expression> init(OUT_LAYERS * 2);
    for (unsigned i = 0; i < OUT_LAYERS; ++i) {
      init[i] = pickrange(c0, i * LSTM_DIM, i * LSTM_DIM + LSTM_DIM);
      init[i + OUT_LAYERS] = tanh(init[i]);
    }
    //cerr << "Milestone 5" << endl;
    outbuilder.new_graph(cg);  // reset RNN builder for new graph
    outbuilder.start_new_sequence(init);
    //cerr << "Milestone 6 " << endl;
    vector<Expression> errs(toks_target.size() + 1);
    we_y->new_graph(cg);
    outbuilder.add_input(we_y->embed(kSOS2));
    cfsm->new_graph(cg);
    //cerr << "Milestone 7" << endl;
    for (unsigned t = 0; t < slen; ++t) {
      Expression h_t = outbuilder.back();
      if (apply_dropout) h_t = dropout(h_t, DROPOUT);
      //Expression u_t = affine_transform({outbias, enc2out, h_t});
      //errs[t] = pickneglogsoftmax(u_t, toks[t]);
      //cerr << "Here 1" << endl;
      //cerr << "Dimension of h_t: " << as_vector(cg.incremental_forward()).size() << endl;
      errs[t] = cfsm->neg_log_softmax(h_t, toks_target[t]);
      //cerr << "Here 2" << endl;
      outbuilder.add_input(we_y->embed(toks_target[t]));
    }
    //cerr << "Milestone 8" << endl;
    Expression h_last = outbuilder.back();
    if (apply_dropout) h_last = dropout(h_last, DROPOUT);
    //Expression u_last = affine_transform({outbias, enc2out, h_last});
    errs.back() = cfsm->neg_log_softmax(h_last, kEOS2);
    sum(errs);
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;

  Model model;
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  if (!conf.count("clusters")) {
    cerr << "Please specify cluster information with the --clusters option" << endl;
    exit(1);
  }
  LSTM_DIM = conf["lstm_dim"].as<unsigned>();
  if (conf.count("clusters"))
    cfsm = new ClassFactoredSoftmaxBuilder(LSTM_DIM, conf["clusters"].as<string>(), &yd, &model);
  IN_LAYERS = conf["in_layers"].as<unsigned>();
  OUT_LAYERS = conf["out_layers"].as<unsigned>();
  CHAR_DIM = conf["char_dim"].as<unsigned>();
  kSOS = xd.Convert("<s>");
  kSOS2 = yd.Convert("<s>");
  kEOS = xd.Convert("</s>");
  kEOS2 = yd.Convert("</s>");
  vector<vector<int>> dev_x, test_x, training_x;
  vector<vector<int>> dev_y, test_y, training_y;
  string line;
  string trainingf = conf["train_1"].as<string>();
  cerr << "Reading training data 1 from " << trainingf << "...\n";

  {
    ifstream in(trainingf);
    assert(in);
    vector<int> x;
    string line;
    while(getline(in, line)) {
      x = ReadSentence(line, &xd);
      training_x.push_back(x);
    }
  }
  xd.Freeze(); // no new word types allowed
  trainingf = conf["train_2"].as<string>();
  cerr << "Reading training data 2 from " << trainingf << "...\n";
  {
    ifstream in(trainingf);
    assert(in);
    vector<int> x;
    string line;
    while(getline(in, line)) {
      x = ReadSentence(line, &yd);
      training_y.push_back(x);
    }
  }
  yd.Freeze(); // no new word types allowed
  assert (training_x.size() == training_y.size());
  VOCAB_SIZE = xd.size();
  cerr << "Source side statistics: " << endl;
  cerr << "  instances: " << training_x.size() << endl;
  cerr << "    # types: " << VOCAB_SIZE << endl;
  cerr << "----------------------------" << endl;

  VOCAB_SIZE = yd.size();
  cerr << "Target side statistics: " << endl;
  cerr << "  instances: " << training_y.size() << endl;
  cerr << "    # types: " << VOCAB_SIZE << endl;
  cerr << "----------------------------" << endl;

  if (conf.count("dev_1") || conf.count("dev_2")) {
    assert (conf.count("dev_1") && conf.count("dev_2"));
    string devf = conf["dev_1"].as<string>();
    cerr << "Reading dev 1 data from " << devf << "...\n";
    ifstream in(devf);
    assert(in);
    vector<int> x;
    string line;
    while(getline(in, line)) {
      x = ReadSentence(line, &xd);
      dev_x.push_back(x);
    }

    devf = conf["dev_2"].as<string>();
    cerr << "Reading dev 2 data from " << devf << "...\n";
    ifstream in2(devf);
    assert(in2);
    x.clear();
    while(getline(in2, line)) {
      x = ReadSentence(line, &yd);
      dev_y.push_back(x);
    }
  }
  assert (dev_x.size() == dev_y.size());
  if (conf.count("test_1") || conf.count("test_2")) {
    assert (conf.count("test_1") && conf.count("test_2"));
    string testf = conf["test_1"].as<string>();
    cerr << "Reading test data 1 from " << testf << "...\n";
    ifstream in(testf);
    assert(in);
    vector<int> x;
    string line;
    while(getline(in, line)) {
      x = ReadSentence(line, &xd);
      test_x.push_back(x);
    }
    testf = conf["test_2"].as<string>();
    cerr << "Reading test data 2 from " << testf << "...\n";
    ifstream in2(testf);
    assert (in2);
    x.clear();
    while(getline(in2, line)) {
      x = ReadSentence(line, &yd);
      test_y.push_back(x);
    }
  assert (test_x.size() == test_y.size());
  }
  Trainer* sgd = new AdamTrainer(&model);
  sgd->eta_decay = 0.08;
  WordEmbed* we_x = new LookupWordEmbed(CHAR_DIM, model, xd);
  WordEmbed* we_y = new LookupWordEmbed(CHAR_DIM, model, yd);
  AEModel<LSTMBuilder> lm(model, we_x, we_y);
  if (conf.count("model")) {
    string fname = conf["model"].as<string>();
    cerr << "Reading model from " << fname << " ...\n";
    ifstream in(fname);
    boost::archive::binary_iarchive ia(in);
    ia >> model;
  }

  if (conf.count("learn")) {
    ostringstream os;
    os << "ae"
       << '_' << IN_LAYERS
       << '_' << OUT_LAYERS
       << '_' << CHAR_DIM
       << '_' << LSTM_DIM
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;

    unsigned report_every_i = 100;
    unsigned dev_every_i_reports = 25;
    unsigned si = training_x.size();
    vector<unsigned> order(training_x.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    double tsize = 0;
    while(1) {
      Timer iteration("completed in");
      double loss = 0;
      unsigned ttags = 0;
      double correct = 0;
      for (unsigned i = 0; i < report_every_i; ++i) {
        if (si == training_x.size()) {
          si = 0;
          if (first) { first = false; } else { sgd->update_epoch(); }
          cerr << "**SHUFFLE\n";
          shuffle(order.begin(), order.end(), *rndeng);
        }
        // build graph for this instance
        ComputationGraph cg;
        const auto& sent = training_x[order[si]];
        const auto& sent_2 = training_y[order[si]];
        //assert (sent.size() == sent_2.size());
        //cerr << "Current sentence: " << order[si] <<  " sent1.size(): " << sent.size() << " sent2.size(): " << sent_2.size() << endl;
        //assert (sent != sent_2);
        if (sent == sent_2) cerr << "Potential mistake! The dictionary indices for the first and second sentences are the same" << endl;
        ++si;
        ttags += sent_2.size();
        tsize += 1;
        //cerr << "Milestone 1" << endl;
        lm.BuildGraph(sent, sent_2, cg, DROPOUT > 0);
        //cerr << "Milestone 2" << endl;
        loss += as_scalar(cg.incremental_forward());
        cg.backward();
        sgd->update(1.0);
      }
      sgd->status();
      cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";
      // show score on dev data?
      report++;
      if (report % dev_every_i_reports == 0) {
        double dloss = 0;
        unsigned dtags = 0;
        double dcorr = 0;
        for (unsigned di = 0; di < dev_x.size(); ++di) {
          const auto& x = dev_x[di];
          const auto& y = dev_y[di];
          //assert (x.size() == y.size() && x != y);
          ComputationGraph cg;
          lm.BuildGraph(x, y, cg, false);
          dloss += as_scalar(cg.incremental_forward());
          dtags += y.size();
        }
        if (dloss < best) {
          cerr << endl;
          cerr << "Current dev performance exceeds previous best, saving model" << endl;
          best = dloss;
          ofstream out(fname);
          boost::archive::binary_oarchive oa(out);
          oa << model;
        }
        cerr << "\n***DEV [epoch=" << (tsize / (double)training_x.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / dtags) << ' ';
      }
    }
  } // should we train?
  if (conf.count("test_1") || conf.count("test_2")) {
    assert (conf.count("test_1") && conf.count("test_2"));
    vector<float> v;
    cerr << "Testing part 1" << endl;
    cerr << "------------------------------------" << endl;
    cerr << "Encoding the source sentence" << endl;
    assert (test_x.size() == test_y.size());
    int idx = -1;
    for (const auto& x : test_x) {
      ++idx;
      ComputationGraph cg;
      Expression enc = lm.EmbedBiLSTM(cg, x, false);
      v = as_vector(cg.incremental_forward());
      for (auto c : x) {
        cout << xd.Convert(c) << ' ';
      }
      cout << "|||";
      int ctr_dim = 0;
      for (auto f : v) {
        cout << ' ' << f;
        ctr_dim++;
      }
      cout << " ||| ";
      assert (ctr_dim == (LSTM_DIM * 2));
      const auto& y = test_y[idx];
      for (auto c : y) {
        cout << yd.Convert(c) << " ";
      }
      lm.BuildGraph(x, test_y[idx], cg, false);
      double sum_loss = as_scalar(cg.incremental_forward()); 
      cout << " ||| " << sum_loss / ((double) (test_y[idx].size() + 1))  <<endl;
    }
   cerr << "------------------------------------" << endl;
   cerr << "Testing part 2" << endl;
   cerr << "Actually decoding from the system" << endl;
   assert (test_x.size() == test_y.size());
   for (int i = 0; i < test_x.size(); ++i) {
          const auto& x = test_x[i];
          const auto& y = test_y[i];
          ComputationGraph cg;
          Expression enc = lm.EmbedBiLSTM(cg, x, false);
          Expression enc2dec = parameter(cg, lm.p_enc2dec);
          Expression decbias = parameter(cg, lm.p_decbias);
          Expression c0 = affine_transform({decbias, enc2dec, enc});
          vector<Expression> init(OUT_LAYERS * 2);
          for (unsigned i = 0; i < OUT_LAYERS; ++i) {
            init[i] = pickrange(c0, i * LSTM_DIM, i * LSTM_DIM + LSTM_DIM);
            init[i + OUT_LAYERS] = tanh(init[i]);
          }
          //cerr << "Milestone 5" << endl;
          lm.outbuilder.new_graph(cg);  // reset RNN builder for new graph
          lm.outbuilder.start_new_sequence(init);
          //cerr << "Milestone 6 " << endl;
          we_y->new_graph(cg);
          lm.outbuilder.add_input(we_y->embed(kSOS2));
          cfsm->new_graph(cg);
          Expression h_t = lm.outbuilder.back();
          double total_nll = 0.0f;
          double best = 9e+99;
          int len = 0;
          int best_idx = -1;
          for (int key = 0; key < yd.size(); ++key) {
            Expression curr_neg_log_softmax = cfsm->neg_log_softmax(h_t, key);
            double curr_score = as_scalar(cg.incremental_forward());
            if (curr_score < best) {
		best = curr_score;
 		best_idx = key;
 	    } 
          }
          total_nll += best;
          len += 1;
          assert (best_idx != -1);
          while (best_idx != kEOS2) {
            cout << yd.Convert(best_idx) << " ";
            lm.outbuilder.add_input(we_y->embed(best_idx));
            Expression h_t = lm.outbuilder.back();
            best = 9e+99; //restart
            best_idx = -1; //restart
            for (int key = 0; key < yd.size(); ++key) {
              Expression curr_neg_log_softmax = cfsm->neg_log_softmax(h_t, key);
              double curr_score = as_scalar(cg.incremental_forward());
              if (curr_score < best) {
                  best = curr_score;
                  best_idx = key;
              }
            }
          total_nll += best;
          len += 1;
          assert (best_idx != -1);
          }
          cout << " ||| " << total_nll / ((double) len) << endl;
    }
  }
}

