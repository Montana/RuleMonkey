// Function settle order (issue #111).
//
// RM settles its function table whenever the state it reads moves, and reads
// every function's value from it.  It used to settle in declaration order,
// which BNG2 does not promise is a dependency order: BNG2 2.9.3's writeXML
// moves only a function's direct references ahead of it, so a chain declared
// top down comes out of order, and a tfun() index is not a reference at all.
// A function settled ahead of one it read got that one's stale value.
//
// Each fixture here is written by BNG2 2.9.3 from the .bngl beside it, its
// chain out of order exactly as BNG2 writes it, and has a closed-form
// expected trajectory.  A table indexed through a function is pinned in
// time_dependent_rate_test.  A model whose functions read each other in a
// cycle has no value to settle to, and is refused.

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

std::size_t obs_index(const rulemonkey::Result& r, const std::string& name) {
  for (std::size_t i = 0; i < r.observable_names.size(); ++i)
    if (r.observable_names[i] == name)
      return i;
  throw std::runtime_error("observable not found: " + name);
}

// The ensemble mean of `obs` at each sample against `mean(t)`, within four
// standard errors, `sd(t)` being one replicate's standard deviation.
void check_mean(const std::string& name, const std::string& xml, const std::string& obs,
                double t_end, int n_points, const std::function<double(double)>& mean,
                const std::function<double(double)>& sd) {
  std::vector<double> sum(static_cast<std::size_t>(n_points) + 1, 0.0);
  std::vector<double> time;
  for (int rep = 0; rep < kReps; ++rep) {
    rulemonkey::RuleMonkeySimulator sim(xml);
    auto const r =
        sim.run({0.0, t_end, n_points}, std::uint64_t{2000} + static_cast<std::uint64_t>(rep));
    auto const& col = r.observable_data[obs_index(r, obs)];
    for (std::size_t k = 0; k < sum.size(); ++k)
      sum[k] += col[k];
    time = r.time;
  }
  std::string const label = name + ": mean " + obs;
  for (std::size_t k = 0; k < sum.size(); ++k) {
    double const t = time[k];
    double const got = sum[k] / kReps;
    double const expect = mean(t);
    double const tol = 4.0 * sd(t) / std::sqrt(static_cast<double>(kReps));
    std::fprintf(stderr, "  %-6s t=%.2f  mean=%8.3f  expected=%8.3f  tol=%.3f\n", name.c_str(), t,
                 got, expect, tol);
    check(std::fabs(got - expect) <= tol, label + "(" + std::to_string(t) +
                                              ") = " + std::to_string(got) + ", expected " +
                                              std::to_string(expect));
  }
}

// Every sample reports each function at its settled value.
void check_functions(const std::string& xml,
                     const std::vector<std::pair<std::string, double>>& expect) {
  rulemonkey::RuleMonkeySimulator sim(xml);
  auto const r = sim.run({0.0, 1.0, 4}, 7);
  for (const auto& [fn, value] : expect) {
    std::size_t fi = r.function_names.size();
    for (std::size_t i = 0; i < r.function_names.size(); ++i)
      if (r.function_names[i] == fn)
        fi = i;
    check(fi < r.function_names.size(), fn + "(): output function not found");
    if (fi == r.function_names.size())
      continue;
    for (std::size_t k = 0; k < r.time.size(); ++k)
      check(r.function_data[fi][k] == value, fn + "() at sample t=" + std::to_string(r.time[k]) +
                                                 " read " + std::to_string(r.function_data[fi][k]) +
                                                 ", expected " + std::to_string(value));
  }
}

// `xml` with `from` replaced by `to` is refused for a function that depends
// on itself.
void check_cycle_refused(const std::string& xml, const std::string& from, const std::string& to) {
  std::ifstream const in(xml);
  std::stringstream text;
  text << in.rdbuf();
  std::string body = text.str();
  auto const pos = body.find(from);
  check(pos != std::string::npos, "cycle: no '" + from + "' in " + xml);
  if (pos == std::string::npos)
    return;
  body.replace(pos, from.size(), to);

  // Unique per run, so two builds testing at once do not share the file.
  auto const tmp =
      std::filesystem::temp_directory_path() /
      ("rm_function_settle_order_test." +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".xml");
  std::ofstream(tmp) << body;
  std::string err;
  try {
    rulemonkey::RuleMonkeySimulator sim(tmp.string());
    sim.run({0.0, 1.0, 1}, 7);
  } catch (const std::exception& e) {
    err = e.what();
  }
  std::filesystem::remove(tmp);
  check(err.find("depends on itself") != std::string::npos,
        "cycle: '" + to + "' was not refused (" + err + ")");
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::fprintf(stderr, "Usage: function_settle_order_test <settle_chain_model.xml> "
                         "<settle_local_chain_model.xml>\n");
    return 2;
  }
  const std::string chain_xml = argv[1];
  const std::string local_xml = argv[2];

  try {
    // a() = b() = 2*c() = 20: B(t) is Poisson with mean 20 t.
    check_mean(
        "chain", chain_xml, "Btot", 5.0, 5, [](double t) { return 20.0 * t; },
        [](double t) { return std::sqrt(20.0 * t); });
    check_functions(chain_xml, {{"a", 20.0}, {"b", 20.0}, {"c", 10.0}});

    // Each of 100 A(s~0) flips at rate 1: A0(t) is binomial(100, exp(-t)).
    check_mean(
        "local", local_xml, "A0", 2.0, 4, [](double t) { return 100.0 * std::exp(-t); },
        [](double t) { return std::sqrt(100.0 * std::exp(-t) * (1.0 - std::exp(-t))); });

    // c() = k closes the chain; reading a() or itself instead makes a cycle.
    check_cycle_refused(chain_xml, "<Expression> k </Expression>",
                        "<Expression> a() </Expression>");
    check_cycle_refused(chain_xml, "<Expression> k </Expression>",
                        "<Expression> 1+c() </Expression>");
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
