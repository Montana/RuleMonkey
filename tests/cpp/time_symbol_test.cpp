// A model symbol named `time` or `t` is that symbol, not the clock (issue #87).
//
// BNG2 rejects a parameter named `time` but accepts an observable named
// `time`.  The bare word is then the observable and the call form `time()` is
// still the clock.  RM bound `time` to the clock and skipped it when binding
// the observable's slot, so a bare `time` in a rate law silently read the
// clock: `k*(time+1)` with the observable at 5 gave B(2) near 400 instead of
// 1200.  BNG2 leaves `t` free too, but RM aliases a bare `t` to the clock and
// let that alias overwrite a parameter named `t`.
//
// The fixture's A() never changes, so the observable `time` holds 5 and every
// function has a closed form at every sample:
//
//   f() = k*(time+1)    600
//   g() = k*(time()+1)  100 (t+1)
//   h() = 10*t          30, from the parameter t = 3
//
// B and C are each fed by one zero-order rule at f and g, so B(t) and C(t)
// are Poisson with means 600 t and 100 (t + t^2/2).

#include "rulemonkey/simulator.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

constexpr int kReps = 20;
constexpr double kTEnd = 2.0;
constexpr int kPoints = 4; // samples at 0, 0.5, 1, 1.5, 2

std::size_t index_of(const std::vector<std::string>& names, const std::string& name) {
  for (std::size_t i = 0; i < names.size(); ++i)
    if (names[i] == name)
      return i;
  throw std::runtime_error("column not found: " + name);
}

// The function columns at every sample against their closed forms.
void check_functions(const std::string& xml) {
  rulemonkey::RuleMonkeySimulator sim(xml);
  auto const r = sim.run({0.0, kTEnd, kPoints}, 7);
  auto const& fd = r.function_data;
  std::size_t const fi = index_of(r.function_names, "f");
  std::size_t const gi = index_of(r.function_names, "g");
  std::size_t const hi = index_of(r.function_names, "h");
  for (std::size_t k = 0; k < r.time.size(); ++k) {
    double const t = r.time[k];
    std::string const at = " at t=" + std::to_string(t) + " read ";
    check(fd[fi][k] == 600.0, "f() = k*(time+1)" + at + std::to_string(fd[fi][k]) + ", not 600");
    check(fd[gi][k] == 100.0 * (t + 1.0), "g() = k*(time()+1)" + at + std::to_string(fd[gi][k]) +
                                              ", not " + std::to_string(100.0 * (t + 1.0)));
    check(fd[hi][k] == 30.0, "h() = 10*t" + at + std::to_string(fd[hi][k]) + ", not 30");
  }
}

// The mean count at t = 2 against its Poisson mean, within four standard
// errors.  Pins that the rate law reads what the function column reports.
void check_rates(const std::string& xml) {
  double sum_b = 0.0;
  double sum_c = 0.0;
  for (int rep = 0; rep < kReps; ++rep) {
    rulemonkey::RuleMonkeySimulator sim(xml);
    auto const r =
        sim.run({0.0, kTEnd, kPoints}, std::uint64_t{2000} + static_cast<std::uint64_t>(rep));
    sum_b += r.observable_data[index_of(r.observable_names, "Bobs")].back();
    sum_c += r.observable_data[index_of(r.observable_names, "Cobs")].back();
  }
  auto check_mean = [](const std::string& what, double sum, double expect) {
    double const mean = sum / kReps;
    double const tol = 4.0 * std::sqrt(expect / kReps);
    std::fprintf(stderr, "  %s  mean=%8.2f  expected=%8.2f  tol=%.2f\n", what.c_str(), mean, expect,
                 tol);
    check(std::fabs(mean - expect) <= tol, what + " mean " + std::to_string(mean) +
                                               ", expected " + std::to_string(expect));
  };
  check_mean("B(2), rate k*(time+1)  ", sum_b, 600.0 * kTEnd);
  check_mean("C(2), rate k*(time()+1)", sum_c, 100.0 * (kTEnd + (kTEnd * kTEnd / 2.0)));
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s time_symbol_model.xml\n", argv[0]);
    return 2;
  }
  try {
    check_functions(argv[1]);
    check_rates(argv[1]);
  } catch (const std::exception& e) {
    check(false, std::string("threw: ") + e.what());
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "time_symbol_test: all checks passed\n");
  return 0;
}
