#include "ntag/word-embed.h"

#include "cnn/dict.h"
#include "cnn/expr.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

WordEmbed::~WordEmbed() {}

void WordEmbed::new_graph_impl() {}

LookupWordEmbed::LookupWordEmbed(unsigned dim, cnn::Model& m, cnn::Dict& xd) :
    WordEmbed(xd) {
  p_p = m.add_lookup_parameters(xd.size(), {dim});
}

Expression LookupWordEmbed::embed_impl(unsigned x) {
  return lookup(*cg, p_p, x);
}

LookupAndPTEmbed::LookupAndPTEmbed(
    unsigned dim,
    unsigned pretrained_dim,
    cnn::Model& m, cnn::Dict& xd,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained) : WordEmbed(xd) {
  p_p = m.add_lookup_parameters(xd.size(), {dim});
  p_pt2w = m.add_parameters({dim, pretrained_dim});
  p_pt = m.add_lookup_parameters(xd.size(), {pretrained_dim});
  for (auto it : pretrained) {
    p_pt->Initialize(it.first, it.second);
    if (it.first >= has_pt.size()) has_pt.resize(it.first + 1, false);
    has_pt[it.first] = true;
  }
}

void LookupAndPTEmbed::new_graph_impl() {
  pt2w = parameter(*cg, p_pt2w);
}

Expression LookupAndPTEmbed::embed_impl(unsigned x) {
  Expression w = lookup(*cg, p_p, x);
  if (has_pt[x]) {
    Expression pt = const_lookup(*cg, p_pt, x);
    w = affine_transform({w, pt2w, pt});
  }
  return w;
}

