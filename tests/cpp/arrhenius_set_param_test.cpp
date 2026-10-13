// An overridden parameter reaches an Arrhenius rule's rate through the same
// formula as the rate computed at load (issue #107).
//
// The loader prices each rule it expands from an Arrhenius rule twice: as a
// number, which a run uses while no parameter is overridden, and as an
// expression of the model's parameters, which apply_overrides re-resolves
// once one is.  The BNG2-oracle suite drives rm_driver, which overrides
// nothing, so it sees only the number.  This test overrides a parameter to
// the value it already has, which changes nothing in BioNetGen's network but
// sends every rate through its expression, and scores the result against the
// same BNG2 references:
//
//   tests/bng_oracle/xml/arrhenius_formula.xml, phi overridden to 0.3: a
//   reversible rule written either way round, unidirectional rules that make
//   and break a bond, and a context-dependent energy pattern;
//   tests/bng_oracle/xml/arrhenius_rt.xml, RT overridden to 2.5: energies in
//   kJ/mol, divided by the model's own RT.
//
// Each observable's mean over the seeds must fall within 6 standard errors
// of the reference at t = 500, as check.py scores the suite.

#include "rulemonkey/simulator.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    ++g_failures;
  }
}

// The last row of a reference TSV, by column name.
std::map<std::string, double> last_row(const std::string& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  std::string line;
  std::vector<std::string> header;
  std::vector<double> row;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream ss(line);
    std::string cell;
    std::vector<std::string> cells;
    while (std::getline(ss, cell, '\t'))
      cells.push_back(cell);
    if (header.empty()) {
      header = cells;
      continue;
    }
    row.clear();
    for (const auto& c : cells)
      row.push_back(std::stod(c));
  }
  std::map<std::string, double> out;
  for (size_t i = 0; i < header.size() && i < row.size(); ++i)
    out[header[i]] = row[i];
  return out;
}

void score(const std::string& xml, const std::string& ref, const std::string& param, double value) {
  constexpr int kSeeds = 24;
  constexpr double kTEnd = 500.0;
  auto const expected = last_row(ref);
  std::map<std::string, std::vector<double>> finals;
  for (int seed = 1; seed <= kSeeds; ++seed) {
    rulemonkey::RuleMonkeySimulator sim(xml);
    sim.set_param(param, value);
    auto r = sim.run(rulemonkey::TimeSpec{0.0, kTEnd, 2}, static_cast<std::uint64_t>(seed));
    for (size_t i = 0; i < r.observable_names.size(); ++i)
      finals[r.observable_names[i]].push_back(r.observable_data[i].back());
  }
  for (const auto& [name, vals] : finals) {
    auto it = expected.find(name);
    std::string where = xml;
    where += " with ";
    where += param;
    where += " overridden: ";
    where += name;
    check(it != expected.end(), where + " has no reference column");
    if (it == expected.end())
      continue;
    double mean = 0;
    for (double const v : vals)
      mean += v;
    mean /= static_cast<double>(vals.size());
    double var = 0;
    for (double const v : vals)
      var += (v - mean) * (v - mean);
    double const se =
        std::sqrt(var / static_cast<double>(vals.size() - 1) / static_cast<double>(vals.size()));
    double const z = std::abs(mean - it->second) / (se > 0 ? se : 1e-12);
    where += " = ";
    where += std::to_string(mean);
    where += " +- ";
    where += std::to_string(se);
    where += ", BNG2 ";
    where += std::to_string(it->second);
    where += " (z ";
    where += std::to_string(z);
    where += ")";
    check(z < 6.0, where);
  }
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: arrhenius_set_param_test <tests/bng_oracle dir>\n");
    return 2;
  }
  std::string const dir = argv[1];
  try {
    score(dir + "/xml/arrhenius_formula.xml", dir + "/reference/arrhenius_formula.mean.tsv", "phi",
          0.3);
    score(dir + "/xml/arrhenius_rt.xml", dir + "/reference/arrhenius_rt.mean.tsv", "RT", 2.5);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures == 0)
    std::fprintf(stdout, "arrhenius_set_param_test: OK\n");
  return g_failures == 0 ? 0 : 1;
}
