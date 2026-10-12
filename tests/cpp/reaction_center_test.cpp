// A rule fires once per distinct REACTION CENTER, wherever its pattern
// lists the transformed molecule (issue #88).
//
// BioNetGen counts a rule's reactions by their reaction center — the
// components whose state or bond the rule changes and the molecules it
// deletes.  Two matches of a reactant pattern that put the center in the same
// place are one reaction however they differ in untransformed context
// (RxnRule::filter_identical_by_rxn_center in BNG2).  RuleMonkey counted a
// slot per seed molecule, seeded every slot on its FIRST pattern molecule,
// and collapsed matches only on that seed's own center components.  So:
//
//   * `L(r!1,r!2).R(l!1,s~0).R(l!2) -> ...` fired 2x on an R linked through
//     two L, because each L was a seed that reached the same center, while
//     the same graph written R first fired 1x;
//   * a center spanning two molecules was collapsed on one of them, merging
//     two distinct reactions into one (1x where BNG2 has 2x);
//   * the collapse kept the first seed embedding of each group whether or
//     not it reached a whole match, so a rule could go inert;
//   * a `.`-joined context molecule with two copies in the complex, and a
//     deleted molecule with symmetric components — alone, or reaching its
//     partner through either arm — each counted 2x;
//   * a per-molecule local function tagged on a context molecule was
//     evaluated at the seed instead of the tag, where it read 0.
//
// reaction_center_model.bngl holds one independent pool per case.  Every
// instance in every pool fires at k and nothing else touches the pool, so a
// pool's unconverted count at t is exactly Binomial(N, exp(-m*k*t)) for the
// multiplicity m BNG2's generate_network() gives it — m = 2 for TwoCenters,
// m = 1 everywhere else.  TwoCentersA counts which of TwoCenters' two
// centers was drawn: they are reached through two complete matches and one,
// and each must still be drawn half the time.

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

double final_value(const rulemonkey::Result& r, const std::string& name) {
  for (size_t i = 0; i < r.observable_names.size(); ++i)
    if (r.observable_names[i] == name)
      return r.observable_data[i].back();
  throw std::runtime_error("observable not found: " + name);
}

struct Pool {
  const char* observable;
  double p; // per-instance probability of still counting at t_end
};

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: reaction_center_test <reaction_center_model.xml>\n");
    return 2;
  }

  constexpr double kN = 2000.0;
  constexpr double kRate = 1.0;
  constexpr double kTEnd = 0.5;
  constexpr int kReps = 8;
  double const once = std::exp(-kRate * kTEnd);        // m = 1 survivor
  double const twice = std::exp(-2.0 * kRate * kTEnd); // m = 2 survivor

  std::vector<Pool> const pools = {
      {"ContextFirst", once},
      {"TransformedFirst", once},
      {"ContextFirstHub", once},
      {"TwoCenters", twice},
      // One of the two centers drawn per conversion, each half the time.
      {"TwoCentersA", (1.0 - twice) / 2.0},
      {"DeadEndFirst", once},
      {"DotContext", once},
      {"DeleteSymmetric", once},
      {"DeleteBoundL", once},
      {"DeleteOneOfTwo", twice},
      {"PartnerBranch", once},
      {"Bimolecular", once},
      {"Nary", once},
      {"TagOnContext", once},
  };

  try {
    std::vector<double> sums(pools.size(), 0.0);
    for (int rep = 0; rep < kReps; ++rep) {
      rulemonkey::RuleMonkeySimulator sim(argv[1]);
      auto r =
          sim.run({0.0, kTEnd, 2}, /*seed=*/std::uint64_t{8800} + static_cast<std::uint64_t>(rep));
      for (size_t i = 0; i < pools.size(); ++i)
        sums[i] += final_value(r, pools[i].observable);
      // The tag's observable is what makes TagOnContext's rate k; it must
      // not move, or the closed form does not apply.
      check(final_value(r, "LtW") == kN, "fixture must hold Lt(w~1) at N");
    }
    for (size_t i = 0; i < pools.size(); ++i) {
      double const p = pools[i].p;
      double const expected = kN * p;
      double const mean = sums[i] / kReps;
      // Binomial per replicate; 5 sigma of the mean is ~40 counts.  Every
      // defect this pins moves its pool by 400 or more.
      double const tol = 5.0 * std::sqrt(kN * p * (1.0 - p) / kReps);
      check(std::abs(mean - expected) < tol,
            std::string(pools[i].observable) + ": expected " + std::to_string(expected) +
                " +- " + std::to_string(tol) + ", observed " + std::to_string(mean));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    ++g_failures;
  }

  if (g_failures == 0)
    std::fprintf(stdout, "reaction_center_test: OK\n");
  return g_failures == 0 ? 0 : 1;
}
