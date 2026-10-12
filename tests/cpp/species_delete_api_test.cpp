// A whole-species delete counts once per matching complex — including on a
// complex the model never makes but the API adds.
//
// BNG2 makes the species the reaction center of `A() -> 0`, so an A.A dimer
// is one reaction at k, not two.  RuleMonkey puts such a slot on the
// per-complex count only where the slot's molecule type can share a
// complex; a type that is a monomer for the whole run already has one match
// per complex, and the tally would only cost it a wide per-molecule row and
// per-event upkeep.  Whether a type can share a complex is read off the
// model at load — and add_species is the one way round that, so it has to
// refresh the answer.
//
// The fixture's A is a monomer at load.  The test adds N dimers through the
// API and runs.  Every A complex, monomer or dimer, is deleted at k, so each
// A molecule survives with probability exp(-k t) and the molecule count is
// Binomial-distributed around 3N exp(-k t).  Left on the per-molecule count,
// the dimers would go at 2k and the count would sit near
// N exp(-k t) + 2N exp(-2k t), 26% lower at k t = 0.5.

#include "rulemonkey/simulator.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    ++g_failures;
  }
}

double final_value(const rulemonkey::Result& r, const std::string& name) {
  for (size_t i = 0; i < r.observable_names.size(); ++i)
    if (r.observable_names[i] == name)
      return r.observable_data[i].back();
  throw std::runtime_error("observable not found: " + name);
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: species_delete_api_test <species_delete_api_model.xml>\n");
    return 2;
  }

  constexpr double kN = 2000.0;
  constexpr double kTEnd = 0.5; // k = 1
  constexpr int kReps = 8;
  double const p = std::exp(-kTEnd);

  try {
    double sum = 0.0;
    for (int rep = 0; rep < kReps; ++rep) {
      rulemonkey::RuleMonkeySimulator sim(argv[1]);
      sim.initialize(std::uint64_t{500} + static_cast<std::uint64_t>(rep));
      sim.add_species("A(b!1).A(b!1)", static_cast<int>(kN));
      auto r = sim.simulate(0.0, kTEnd, 2);
      sum += final_value(r, "A_tot");
    }
    double const mean = sum / kReps;
    // A_tot = monomers + 2 * dimers, each count Binomial(N, p).
    double const expected = 3.0 * kN * p;
    double const tol = 5.0 * std::sqrt(5.0 * kN * p * (1.0 - p) / kReps);
    check(std::abs(mean - expected) < tol,
          "A.A dimers added through the API must be deleted once per complex: expected " +
              std::to_string(expected) + " +- " + std::to_string(tol) + ", observed " +
              std::to_string(mean));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    ++g_failures;
  }

  if (g_failures == 0)
    std::fprintf(stdout, "species_delete_api_test: OK\n");
  return g_failures == 0 ? 0 : 1;
}
