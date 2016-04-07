#ifndef NTAG_WORD_EMBED_H
#define NTAG_WORD_EMBED_H

#include <vector>
#include <unordered_map>

#include "cnn/expr.h"

namespace cnn {
  struct ComputationGraph;
  struct Dict;
  struct LookupParameters;
  struct Model;
}

// there are lots of ways to embed words, this lets us have
// a common interface to all of them
//   - call new_graph(cg) before asking for embeddings on a new graph
//   - then call embed(x) as many times as you want
struct WordEmbed {
  WordEmbed(const cnn::Dict& d) : d(d) {}
  virtual ~WordEmbed();

  void new_graph(cnn::ComputationGraph& g) {
    cg = &g;
    cache.clear();
    new_graph_impl();
  }
  virtual void new_graph_impl();
  const cnn::expr::Expression& embed(unsigned x) {
    auto& e = cache[x];
    if (!e.pg) e = embed_impl(x);
    return e;
  }
  virtual cnn::expr::Expression embed_impl(unsigned x) = 0;
 protected:
  const cnn::Dict& d;
  cnn::ComputationGraph* cg;
 private:
  std::unordered_map<unsigned, cnn::expr::Expression> cache;
};

// simple stupid lookup tables
struct LookupWordEmbed : public WordEmbed {
  LookupWordEmbed(unsigned dim, cnn::Model& m, cnn::Dict& xd);
  cnn::expr::Expression embed_impl(unsigned x) override;
  cnn::LookupParameters* p_p;
};

// W*pt(x) + w(x) where W and w(x) are learned and fixed pt(x) is a pretrained embedding or 0 if its not in the pretrained vocab
struct LookupAndPTEmbed : public WordEmbed {
  LookupAndPTEmbed(
    unsigned dim,
    unsigned pretrained_dim,
    cnn::Model& m, cnn::Dict& xd,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);
  void new_graph_impl() override;
  cnn::expr::Expression embed_impl(unsigned x) override;
  cnn::LookupParameters* p_p;
  cnn::LookupParameters* p_pt;
  cnn::Parameters* p_pt2w;
  cnn::expr::Expression pt2w;
  std::vector<bool> has_pt;
};

// TODO character embeddings, etc

#endif
