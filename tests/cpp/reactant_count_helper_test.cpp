// `reactant_N()` reached through a helper function (issue #59).
//
// The placeholder stands for the match count of the asking rule's Nth
// reactant pattern, so RM binds the rule's counts into it before evaluating
// the rule's rate law.  It bound them for that one evaluation only.  A rate
// law that reads the placeholder through a helper,
//
//     g()  = reactant_1()*kf()
//     A(s~0) -> A(s~1)  2*g() TotalRate      (BNG2 writes _rateLaw1 = 2*g())
//
// reads the helper's slot, which the settle pass had filled with the
// placeholder at zero.  The rate was zero and the rule never fired, silently.
// A helper two rules read held one rule's value, so the other was priced off
// the wrong count; a TFUN whose counter reads the placeholder was not seen to
// read it at all; and the placeholder named as the whole rate law, which has
// no expression, read the rule's static rate of zero.
//
// Each fixture is written by BNG2 2.9.3 from the .bngl beside it, except the
// TFUN one, which is hand-edited since stock BNG2 2.9.3 cannot parse tfun().
// Every rule is a TotalRate state flip, so each molecule flips independently
// at a constant rate and the count left is binomial with a closed-form mean.
// The functions each sample reports are pinned too: a helper reports its
// value against the rule that owns it, a rate law against its own rule.
// A local rate law reaching the placeholder through a helper stays refused.

#include "rulemonkey/simulator.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
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

constexpr int kReps = 200;
constexpr double kK = 1e-2;

std::size_t obs_index(const rulemonkey::Result& r, const std::string& name) {
  for (std::size_t i = 0; i < r.observable_names.size(); ++i)
    if (r.observable_names[i] == name)
      return i;
  throw std::runtime_error("observable not found: " + name);
}

std::size_t fn_index(const rulemonkey::Result& r, const std::string& name) {
  for (std::size_t i = 0; i < r.function_names.size(); ++i)
    if (r.function_names[i] == name)
      return i;
  throw std::runtime_error("function not found: " + name);
}

// `n0` molecules each flipping at `rate`: the count left at t is
// binomial(n0, exp(-rate t)).
struct Decay {
  std::string obs;
  double n0;
  double rate;
};

// The ensemble mean of each decay's observable at each sample against its
// closed form, within four standard errors.
void check_means(const std::string& name, const std::string& xml, double t_end, int n_points,
                 const std::vector<Decay>& decays) {
  std::vector<std::vector<double>> sum(
      decays.size(), std::vector<double>(static_cast<std::size_t>(n_points) + 1, 0.0));
  std::vector<double> time;
  for (int rep = 0; rep < kReps; ++rep) {
    rulemonkey::RuleMonkeySimulator sim(xml);
    auto const r =
        sim.run({0.0, t_end, n_points}, std::uint64_t{5960} + static_cast<std::uint64_t>(rep));
    for (std::size_t d = 0; d < decays.size(); ++d) {
      auto const& col = r.observable_data[obs_index(r, decays[d].obs)];
      for (std::size_t k = 0; k < col.size(); ++k)
        sum[d][k] += col[k];
    }
    time = r.time;
  }
  for (std::size_t d = 0; d < decays.size(); ++d) {
    auto const& dc = decays[d];
    for (std::size_t k = 0; k < time.size(); ++k) {
      double const t = time[k];
      double const p = std::exp(-dc.rate * t);
      double const got = sum[d][k] / kReps;
      double const expect = dc.n0 * p;
      double const tol = 4.0 * std::sqrt(dc.n0 * p * (1.0 - p) / kReps);
      std::fprintf(stderr, "  %-14s %-4s t=%5.2f  mean=%8.3f  expected=%8.3f  tol=%.3f\n",
                   name.c_str(), dc.obs.c_str(), t, got, expect, tol);
      check(std::fabs(got - expect) <= tol, name + ": mean " + dc.obs + "(" + std::to_string(t) +
                                                ") = " + std::to_string(got) + ", expected " +
                                                std::to_string(expect));
    }
  }
}

// Every sample reports function `fn` at `value(obs)`, `obs` being that
// sample's observables by name.
using ObsAt = std::function<double(const std::string&)>;
void check_reported(
    const std::string& name, const std::string& xml, double t_end,
    const std::vector<std::pair<std::string, std::function<double(const ObsAt&)>>>& expect) {
  rulemonkey::RuleMonkeySimulator sim(xml);
  auto const r = sim.run({0.0, t_end, 8}, 11);
  for (const auto& [fn, value] : expect) {
    std::size_t const fi = fn_index(r, fn);
    for (std::size_t k = 0; k < r.time.size(); ++k) {
      ObsAt const obs = [&](const std::string& o) { return r.observable_data[obs_index(r, o)][k]; };
      double const want = value(obs);
      double const got = r.function_data[fi][k];
      if (std::fabs(got - want) > 1e-12 * std::fmax(1.0, std::fabs(want))) {
        std::fprintf(stderr, "FAIL: %s: %s() at t=%g reported %.17g, expected %.17g\n",
                     name.c_str(), fn.c_str(), r.time[k], got, want);
        ++g_failures;
      }
    }
  }
}

// `xml` with the `<Function id="moved">` element moved ahead of
// `<Function id="before">`, written to a temporary file.
std::filesystem::path reorder(const std::string& xml, const std::string& moved,
                              const std::string& before) {
  std::ifstream const in(xml);
  std::stringstream text;
  text << in.rdbuf();
  std::string body = text.str();
  std::string const open_moved = "      <Function id=\"" + moved + "\"";
  std::string const close = "      </Function>\n";
  auto const start = body.find(open_moved);
  auto const end = body.find(close, start);
  if (start == std::string::npos || end == std::string::npos)
    throw std::runtime_error("reorder: no function '" + moved + "' in " + xml);
  std::string const element = body.substr(start, end + close.size() - start);
  body.erase(start, element.size());
  auto const at = body.find("      <Function id=\"" + before + "\"");
  if (at == std::string::npos)
    throw std::runtime_error("reorder: no function '" + before + "' in " + xml);
  body.insert(at, element);

  // Unique per run, so two builds testing at once do not share the file.
  auto const tmp =
      std::filesystem::temp_directory_path() /
      ("rm_reactant_count_helper_test." +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".xml");
  std::ofstream(tmp) << body;
  return tmp;
}

void check_refused(const std::string& xml, const std::string& must_contain,
                   const std::string& tag) {
  rulemonkey::RuleMonkeySimulator const sim(xml);
  bool found = false;
  for (const auto& u : sim.unsupported_features())
    if (u.severity == rulemonkey::Severity::Error &&
        u.feature.find(must_contain) != std::string::npos)
      found = true;
  check(found, tag + " is refused at load with an Error");
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::fprintf(stderr, "Usage: reactant_count_helper_test <helper.xml> <shared_helper.xml> "
                         "<bare.xml> <tfun_counter.xml> <local_helper.xml>\n");
    return 2;
  }
  const std::string helper_xml = argv[1];
  const std::string shared_xml = argv[2];
  const std::string bare_xml = argv[3];
  const std::string tfun_xml = argv[4];
  const std::string local_xml = argv[5];

  try {
    // _rateLaw1 = 2*g(), g() = reactant_1()*kf(): each A(s~0) flips at 2k.
    check_means("helper", helper_xml, 40.0, 4, {{"Aoff", 100.0, 2.0 * kK}});
    check_reported("helper", helper_xml, 40.0,
                   {{"g", [](const ObsAt& o) { return o("Aoff") * kK; }},
                    {"_rateLaw1", [](const ObsAt& o) { return 2.0 * (o("Aoff") * kK); }}});

    // The same model with the rate law declared ahead of the helper it
    // reads.  RM settled in declaration order before issue #111, and this
    // order then worked by accident: the rate law's binding was left in the
    // placeholder when g() came to be settled.
    {
      auto const tmp = reorder(helper_xml, "_rateLaw1", "g");
      check_means("reordered", tmp.string(), 40.0, 4, {{"Aoff", 100.0, 2.0 * kK}});
      std::filesystem::remove(tmp);
    }

    // g() is the A rule's whole rate law and read by the B rule's
    // _rateLaw1 = 3*g(): each A flips at k and each B at 3k.  g() reports its
    // value against the A rule, which owns it.
    check_means("shared", shared_xml, 40.0, 4, {{"Aoff", 100.0, kK}, {"Boff", 50.0, 3.0 * kK}});
    check_reported("shared", shared_xml, 40.0,
                   {{"g", [](const ObsAt& o) { return o("Aoff") * kK; }},
                    {"_rateLaw1", [](const ObsAt& o) { return 3.0 * (o("Boff") * kK); }}});

    // reactant_1() as the whole rate law: each A(s~0) flips at rate 1.
    check_means("bare", bare_xml, 2.0, 4, {{"Aoff", 100.0, 1.0}});

    // f() = k*tfun over cnt() = reactant_1(), an identity table: the A rule
    // names f() as its rate law and the B rule reads 2*f().
    check_means("tfun_counter", tfun_xml, 40.0, 4, {{"Aoff", 100.0, kK}, {"Boff", 50.0, 2.0 * kK}});
    check_reported("tfun_counter", tfun_xml, 40.0,
                   {{"cnt", [](const ObsAt& o) { return o("Aoff"); }},
                    {"f", [](const ObsAt& o) { return kK * o("Aoff"); }},
                    {"_rateLaw1", [](const ObsAt& o) { return 2.0 * (kK * o("Boff")); }}});

    check_refused(local_xml, "local (per-instance) function",
                  "a count read through a helper from a local-function rate law");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "PASS\n");
  return 0;
}
