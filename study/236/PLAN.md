# #236 study plan, rev 9d (ideal gas only)

Study plan for issue #236; not for merge. Drafting only; nothing run.

Rev 9d critique -> change (parent 8a7b0eefc7a03c17a58c1413fd2b04359f3e67a9)
- Coordinating bot's Gate 0 cross-check: correct YAML u0_R from the mistaken T=0 interpretation to internal energy/R at the card's Tref, and use Kintera R=8.31446; recompute O3 card values (PROPOSED), G/I1 energy formulas, I2 scale and cloud conditioning. Sources are Kintera 4dc613d thermo_y.cpp:63-67,82, thermo_options.cpp:47-48 and constants.h:6; both Gate 0 EOS paths construct ThermoY. Sign-off restarts on this sha.
- Qualification: Tref is card-configurable, default 300 K, not a universal hard-coded 300 K; the one-time shift applies when offset_zero is false (thermo.hpp:77,84; thermo_y.cpp:63-67,82). The Gate 0 base card sets 300 K. No disagreement with the corrected card-specific energy formula.
- Apply dungeon2 review items (1)-(5): rev 9c parent recorded; heading-only cross-references replaced with specific response/note lines; both earlier decision-fence edits quoted word for word; Larrouturou inline coordinating-bot credit added; Hu's 2x vote explicitly required at every N. The earlier text said 'report it for every listed N' but did not explicitly require every N to pass.
- The four protected regions stay byte-identical to 6f9e0adc. Historical O1/non-ideal values and old control miss sizes are not recomputed claims; affected active numerical references are superseded below. Unless explicitly labelled rev 9d, old history/table line references retain their original revision's numbering.

Rev 9c critique -> change (review of rev 9b; parent 19b337f31d868fb90abbc00c9ba8cd78d6941fdb)
- Rev 9c: separates Wong positivity/order tests, makes Hu's two checks vote, tightens the Larrouturou vote, adds a second carry card and closes the independent rev 9a review leftovers; Sihe Chen's items (Slack 1790887060.769699) and coordinating-bot items (Slack 1790887177.825939) are mapped below. Sign-off restarts on this sha.
- 1 (Sihe Chen): Wong interface uses initial trace density, not a floor, with positivity only; a separate proposed smooth trace card carries the L1 order gate (L123-L124); rev 9b response 7 is corrected.
- 2 (Sihe Chen): Larrouturou CFL and paper/study species mappings are explicit (L125-L126).
- 3a: I1 now targets O3, not a dropped O1 residual (L294).
- 3b: I2 defines its own S_n (L295).
- 3c: Gates 2/3 use 12 z=1 configurations per device and A/G-specific oracles (L399-L400).
- 3d: Gate 4's A-only scope was already closed in rev 9b at L349; retained at L401.
- 3e: actual-clamp versus marked-cell distinction was closed in rev 9b at L140,152-155,374; ghost clamps explicitly count now (L160, L177).
- 3f: Hu N ladder, t=1, fitted order and zero-clamp unlimited reference are explicit (L122, L128).
- 3g: rev 9a changed "Superseded by rev 9: the entire historical A/B decision below is replaced by the 12:58 DECIDE and its linked PROPOSED accuracy threshold; the W->E bug remains its own fix." to "Superseded by rev 9: the entire historical A/B decision below is replaced by the 12:58 DECIDE and its linked rev 9a per-test accuracy criteria; the W->E bug remains its own fix."; rev 9b changed "Superseded by rev 9: the entire historical A/B decision below is replaced by the 12:58 DECIDE and its linked rev 9a per-test accuracy criteria; the W->E bug remains its own fix." to "Superseded by rev 9: the entire historical A/B decision below is replaced by the 12:58 DECIDE and its linked rev 9b per-test accuracy criteria; the W->E bug remains its own fix.". Both edits are now logged word for word; the fence is unchanged in rev 9d (rev 9d L228-L229 and L234).
- 3h: the '7 numerics papers' count remains as written inside the verbatim 12:58 block; the count discrepancy is noted, not edited (rev 9d L229 and L234).
- bot-9c-1: nontrivial 0.7/0.3 initial fractions and their range replace the vacuous [0,1]-based vote; the pure-species paper comparison does not vote (L125-L126).
- bot-9c-2: both Hu order and CFL-0.5 2x L-inf checks vote (L122); the new ladder/time clarification does not change the latter's same-resolution scope.
- bot-9c-3: second proposed carry card, separate-process recipe and run-matrix row added, with inspected source citations (L179-L183).

Rev 9b critique -> change (review of rev 9)
- Rev 9b: oracle-review items B1, B2, 1-11 from the coordinating bot's independent review of rev 9 (Slack, 2026-10-01); Xi->Sihe Chen attribution fix for rev 9a. Sihe Chen's approve on 5f175ed4 is superseded; sign-off round restarts on this sha.

Rev 9a critique -> change (review of e0c1c822)
- rev 9a: per-test DECIDE accuracy criteria replacing the single PROPOSED threshold of rev 9, following Sihe Chen's rev 9 review (SIGN-OFF changes on e0c1c822). Items 1 and 2 as Sihe Chen stated; the Larrouturou bound is written as the single interval [-1e-3, 1+1e-3] and remains PROPOSED for confirmation by the three signers.

Rev 9 critique -> change (review of 05d5e6c)
- Folds in the 11:51 KEEP/DROP list (ideal gas only) and the 12:58 literature-read plan, in that order and verbatim below.
- The 12:58 DECIDE supersedes the A/B decision rule: compare A against G on z=1, with P extended to the CFL ladder.
- The 11:51 ADD of an empty Literature section is superseded by the filled Literature section from the 12:58 post.
- The single accuracy threshold introduced in rev 9 is superseded by the rev 9c per-test DECIDE criteria below.

Earlier revision entries are retained as history; their old scope, gates and A/B decisions are superseded by rev 9 where noted below.

Rev 8b critique -> change (review of db40758)
- 1. L186 and L195 omitted the cp override precondition -> both the gas/cloud czh line and Gate 1b require use_nasa9_cp and use_h2_cp off; eval_uhs.cpp:289-300 overrides intEng otherwise, including torch::where(n9.mask, intEng_nasa, result) at :294 on kintera 4dc613d.
- 2. L210's first sentence conflicted with its ghost-cell exception -> prefix the mass-conservation claim with "In the interior," and keep interior and ghost added mass separate.
- 3. L213 and L220 treated block flags as cell counts and repeated settled CPU evidence -> add a per-cell counter / mask dump in the study build, record interior and ghost mass separately because marks see only the interior, and identify CUDA as the new deciding evidence; #263's CPU run already has 4000 cycles / 0 redos.

Rev 8 critique -> change (review of b301975)
- L187. Gate 1b said every cloud extra reads back czh = 1, but kintera 4dc613d eval_uhs.cpp:225-226 sets czh = 1 only on the vapor_ids slice (dry included) -> cloud czh = 0, czh_ddC = 0, intEng_R = uref_R + T cref_R exactly; cloud slots start at vapor_ids().size() (thermo_y.cpp:78).
- L179. G-AB said czh = 1 for all species -> czh = 1 on gases and 0 on clouds (eval_uhs.cpp:225-226, 247, 268-288); the cloud half drops the z T term from S_n and notes the NASA-9/H2 cp overrides (eval_uhs.cpp:289-300).
- L69. The call_func2 ranges were wrong (utils_dispatch.cpp:47-67, utils_dispatch.cu:42-102) -> kintera 4dc613d call_func2_cpu src/utils/utils_dispatch.cpp:47-69 (registered :110), call_func2_cuda src/utils/utils_dispatch.cu:42-65 (registered :102).
- L202. The mass line did not say where the clamp acts -> clamp_min_(0) at equation_of_state.cpp:327 acts on all of cons (ghost cells and every cloud), while fix_vapor at :321 is interior only, so parentless clouds in ghost cells gain mass without passing through fix_vapor first.
- P. The deciding run_hydro run had no length -> it is 4000 cycles, as in #263.

Rev 7 critique -> change (review of 9ed02b7)
- a. L78 cited moist_mixture.cpp:202, an 842a116 line (at 5eeb9b6 it is `auto mom = ...`) -> it cites conc = V * inv_mu at moist_mixture.cpp:223 and :236 on 5eeb9b6.
- b. L79 cited h *= Rgas*inv_mu at :211, an 842a116 line (at 5eeb9b6 it is `int ny = ...`) -> it cites W->E, which does this at moist_mixture.cpp:184 on 5eeb9b6.
- c. P cited the repair as equation_of_state.cpp:256-327 and left out the vapour pass and the marks -> it cites :253-354 on 5eeb9b6 in step order: parent borrow (:269-285), parentless fix_vapor + clamp_min_(0) (:302-328), vapour pass via columnar fix_vapor with TORCH_CHECK (:330-344), marks (:348-353).
- d. Gate 4 (2) did not define "unlimited face" -> an unlimited face is one where every species' donor theta == 1 (share == 0; flux_positivity.cpp:127-136 on 5eeb9b6).
- e. P's decision rule did not name the devices -> the marked-cells rule applies to the run_hydro run on both CPU and CUDA.
- f. The #263 context implied the run_hydro run measured denormal negatives -> #263's run_hydro row shows "-" for negative species at step start; the denormal negatives were measured only with the refreshed external driver.
- g. Rev 6 line g said L79 and the grid cite 5eeb9b6 lines; the grid did, but L78 (:202) and L79 (:211) still held 842a116 line numbers -> fixed by a and b above.

Rev 6 critique -> change (review of 07fc199)
- a. The rev 5 line (a) misstated what Gate 0 did (rev 4 already recorded mismatches) -> it says Gate 0 built main 3f7ad96, which is not current main.
- b. Gate 4 (2) "formulation A with no extras is bitwise equal" could not pass (main pins a moist-mixture gap test) -> Gate 4 excepts moist_mixture_withholds_no_energy_or_momentum_yet, replaced by expect_carried(off, on), and (2) is bitwise equality on unlimited faces and in the species rows on limited faces, for one forward pass; check (1) is kept.
- c. P described main's repair wrongly (a parented cloud borrows from its parent vapour first; the rows are partial densities) -> P cites equation_of_state.cpp:256-327 and states where mass is added (clamp_min_(0) on a parentless cloud only).
- d. The reproducer's "clamps on nearly every step" held only with the external driver -> it states both drivers: species repairs nearly every step with the external driver, 0 redos with run_hydro.
- e. Gate 0 asked for kintera >= 2.5.13 -> kintera v2.5.15 (4dc613d, kintera main) for both builds, so the bitwise checks reproduce.
- f. P had no decision rule -> the face limiter is the guarantee iff marked cells = 0 on the run_hydro run. Superseded by rev 9b: hits > 0, clamp attribution and mass closure are also required.
- g. L79 and the grid cited lines of 842a116, which no chengcli ref holds -> they cite 5eeb9b6 lines. The mapping found two 842a116 tests that do not exist at 5eeb9b6: moist_mixture_withheld_mass_keeps_its_energy_and_momentum (the grid's MM z=1 lmars cell is now the gap test at 5eeb9b6:351) and moist_mixture_nasa9_h2_enthalpy_matches_internal_plus_pressure (the hand h_dry test; a new test does this).

Rev 5 critique -> change
- a. Gate 0 built main 3f7ad96, which is not current main -> Gate 0 builds current main 5eeb9b6 (v2.10.36, kintera >= 2.5.13); the 3f7ad96 counts are a prior only, and a count mismatch is recorded, not an abort.
- b. Gate 4 used 842a116 as a bitwise oracle -> Gate 4 = full ctest plus two arm-A-only bitwise checks, both against 5eeb9b6 (neither constrains G): (1) ideal-moist fluxes, (2) formulation A with no extras; 842a116 is a candidate only and never the oracle; no rebase comparison.
- c. The non-ideal option's species scope was implicit -> the test-only option registers virial z and u for dry gas and vapour ONLY and leaves every cloud extra unset; this scope is stated in the option text, in O1 and in Gate 1b.
- d. Where species positivity is guaranteed was not studied -> new item P: face flux limiter vs cell clamp, what the cell clamp does to mass, energy and momentum, and what it reports; reproducer is the 2D moist Jupiter CRM with limiter: true that clamps vapour/cloud on nearly every step (#263).

Issue: https://github.com/chengcli/snapy/issues/236

## rev 9: ideal-gas scope and literature-read plan

The following two coordinating-bot posts are preserved verbatim. The 12:58 plan is the later decision where the posts overlap; it supersedes the A/B decision rule. Non-ideal gases remain later work.

### 11:51 KEEP/DROP list (ideal gas only)

```text
KEEP   formulation A as the carry (h_n = u_n + R_n T per gas, h_c = u_c per cloud, + KE); O3 (YAML only, z=1) for
       every species incl. dry; I1 (sum identity, dry included); I2 (locality); the controls C, dropped inv_mu and
       condensate; the use_nasa9_cp / use_h2_cp precondition. Gate 0 (main 5eeb9b6 counts, kintera v2.5.15);
       Gates 2 + 3 on the z=1 cells only: 6 moist-mixture cases x {CPU, CUDA} = 12, plus the 12 ideal-moist
       controls = 24 cells, a skip is a gap; Gate 4 (full ctest + python limiter tests, the two bitwise checks vs
       5eeb9b6, flip moist_mixture_withholds_no_energy_or_momentum_yet to expect_carried). Item P as written
       (CUDA is the deciding run; CPU per-cell mass/energy/momentum table). W->E kg/m3 bug stays its own fix.
DROP   formulation B; the kintera test-only virial option; eval_intEng_R_ddC and its kintera PR; the 12 z!=1
       cells; O1, O2, I3, I4, X, A-NI, G-AB; the A-vs-B runtime gate and the A/B decision rule.
LATER  non-ideal gases (z!=1) get a Limit line in the fix PR and a new upstream issue for human discussion.
ADD    a "Literature" section, empty for now: we are running a literature pass (species-positivity limiters
       and the energy/momentum carried with limited species mass) and will post citations with page numbers.
```

The 11:51 ADD of an empty Literature section is superseded by the filled Literature section from the 12:58 post below.

### 12:58 plan after the literature read (Xi approved: "好，就这样")

```text
FINDING  Our per-species cut (and A, which adds h_n + KE to it) appears in none of the 7 numerics papers read.
         The standard is G: blend the WHOLE face flux vector toward a positivity-safe first-order flux with ONE
         coefficient (Hu, Adams & Shu 2013 JCP 242 eq 12, LF, CFL <= 1/2; Wong et al. 2021 JCP 444 Alg. 4, HLLC,
         CFL <= 0.5; Clayton et al. 2026 Comput. Fluids 317 eq 5.1). Energy and momentum stay consistent by
         construction. Thermo (Satoh 2003 eq 9-10, 41; Bannon 2002 eq 6.1-6.4; Li & Chen 2019 eq 10, 14) supports
         A's weights: gas c_p T + h0, cloud c_l T + h0 (no pV), + KE, one shared reference. Caution: Subbareddy
         et al. 2017 JCP 348 eq 9 gives dE/drho_s at fixed p,u, which is not h_n + KE. No LMARS positivity proof
         was found; every proof needs CFL <= 0.5, and the #263 reproducer runs cfl 0.9.
COMPARE  A (per-species cut + h_n + KE) vs G (common-coefficient blend toward a positivity-safe HLLC/LF flux),
         both on the z=1 scope.
TESTS    (1) the 24 z=1 cells: limited-face energy/momentum residuals, column totals IDN, ICY.., IPR, IVX-IVZ.
         (2) literature tests: Wong 1D interface advection with a 1e-8 partial-density floor (positivity, order);
             Larrouturou 1991 two-species Sod tube (Y overshoot); Hu smooth advection u = 1 + 1e-6 + cos(2 pi x)
             (L-inf error vs CFL).
         (3) P as a CFL ladder: #263 CRM at cfl 0.9 / 0.5 / 0.3, CPU and CUDA: clamped cells, marked cells,
             mass added by the clamp (interior and ghost), per step.
DECIDE   G if it conserves exactly, keeps species >= 0 with no clamp at CFL <= 0.5, and its accuracy loss in (2)
         is acceptable (state the threshold in rev 9); otherwise A. Report the cfl 0.9 result either way.
ADD      a Literature section with these citations (eq/page as above); full notes are in our reading notes.
```

**Per-test DECIDE accuracy criteria (rev 9c):** Apply these to TESTS (2); the verbatim 12:58 DECIDE remains unchanged. The revised timing, diagnostics and G implementation below are PROPOSED pending the restarted three-signer review.

1. Hu, Adams & Shu 2013 smooth advection (u = 1 + 1e-6 + cos(2 pi x)): two checks VOTE under DECIDE: (a) G's observed convergence order within 0.5 of unlimited on the order ladder below; (b) at CFL 0.5 and the same resolution, G's L-inf error <= 2x the unlimited scheme. Measure convergence order separately with dt = 0.5*dx^(5/3) in both arms; G's observed order must be within 0.5 of unlimited. The timestep is per oracle review, to verify against Hu 2013 p.11; no local PDF was found. The original numerical thresholds were accepted by Sihe Chen; the revised timing awaits all three signers. PROPOSED variable convention pending the paper: u_adv is the advected scalar/partial density carrying the 1e-6 offset, while transport velocity is v; confirm whether the paper's u denotes that scalar before implementing, rather than silently interpreting it as Snapy's velocity. Require a nonzero count of limited faces in G, not merely a passing error norm.
2. Wong et al. 2021: separate the interface positivity case from the smooth-order case, following Sihe Chen (Slack 1790887060.769699). Interface (§6.2 / Table 5): a single N=200 piecewise-constant run; 1e-8 is the INITIAL trace partial density, not a floor. Its ONLY gate is every species partial density >= 0 at every cell and step, with no clamp; report the per-cell clamp counter including ghosts, max |p-p0|/p0 and max |u-u0| (u is velocity here). No interface order gate and no L-inf-vs-unlimited gate. Smooth trace advection (§6.1 / Table 4 style) has the separate L1 order vote specified below: G's observed order within 0.5 of the unlimited scheme. These are study criteria, not a claim that the proposed smooth card reproduces Table 4 exactly.
   Smooth trace card (all chosen values PROPOSED): periodic x in [0,1], constant total density rho=1 kg/m3, p0=1e5 Pa, velocity u0=1 m/s; first non-dry gas partial density rho_1(x,0)=1e-8*(1+0.5*sin(2*pi*x)), dry gas rho_2=1-rho_1, no clouds. Gas 1/2 use gamma 1.4/1.67 and molar mass 29/4 g/mol (the retained PROPOSED pair); construct T and energy consistently from that ideal-mixture EOS. N=50,100,200,400,800 uniform cells, final time t=1, CFL=0.5. Initialize exact cell averages and compare at t=1 against the analytically translated cell averages; E1(N)=sum_i dx*|rho_1,i-rho_1,exact,i|. Fit observed order as the least-squares slope of log E1 against log dx over all five N, separately for G and unlimited; report all errors and adjacent-grid orders. A zero/round-off-saturated error makes the fit indeterminate, not an automatic pass. Use the B2 zero-clamp reference and report positivity/clamp and contact diagnostics on this card too. Interface and smooth-order votes both feed DECIDE.
3. Larrouturou 1991 two-species Sod tube (voting-run revision credited to the coordinating bot's 9c review, Slack 1790887177.825939): both A and G receive a PROPOSED DECIDE vote from a nontrivial-mixture run, with Y_1,L=0.7, Y_1,R=0.3 and Y_2,L=0.3, Y_2,R=0.7. Require each Y_n in [min_x Y0_n - 1e-3, max_x Y0_n + 1e-3] (here [0.299,0.701]) at every cell and step. Report beside EACH arm the unlimited maximum excursion max over species, cells and steps of max(0, min_x Y0_n-Y_n, Y_n-max_x Y0_n); report the same excursion for that arm. No order gate; round-off-level tightness is NOT required (Sihe Chen's review); the bound awaits all three signers. Rationale: with two nonnegative partial densities summing to rho, both Y_n already lie in [0,1], so the old [-1e-3,1+1e-3] vote cannot distinguish positivity-preserving arms. Keep Y_1,L=1, Y_1,R=0 as a REPORTED, NON-VOTING comparison against the paper's 1+1.5e-6 overshoot; report max Y_1 and its excursion separately from the nontrivial-mixture vote.
   Larrouturou card for BOTH runs: CFL=0.5 is our PROPOSED choice; the paper (INRIA RR-1080 §5) used CFL=0.75. Paper Y_1 is the initially left gas (Y_L=1, Y_R=0, gamma 1.4), Y_2 the initially right gamma-1.2 gas; paper resolution 101 points, final time t=0.21. No Snapy species mapping was fixed in rev 9b: PROPOSED Y_1 is the first non-dry vapour component (ICY), Y_2 is dry (IDN); no cloud. Retain the rev 9b item-7 PROPOSED study pair gamma 1.4/1.67, molar mass 29/4 g/mol for Y_1/Y_2 in BOTH study runs. This differs from the paper's gamma 1.4/1.2, so the non-voting reported comparison is an adapted case, not a numerical reproduction of its overshoot. PROPOSED finite-volume realization: 101 cells on [0,1], diaphragm x=0.5, final time t=0.21 in paper units; initialize diaphragm-cut cells with the exact volume averages. The dimensionalization of the remaining paper states and boundary conditions is an open implementation prerequisite: set it consistently with the EOS/floors before running, without changing it between the two cards. All parameters are identical between the voting and reported cards except initial fractions. Paper parameters above are attributed to the review, pending direct paper verification.

Hu run specification (rev 9c): N=50,100,200,400,800 (paper's 50-800 ladder, per review), final time t=1. PROPOSED fitting convention: exact cell-average initialization/reference on the periodic unit interval, fit the least-squares slope of log L-inf error versus log dx over all five resolutions, with adjacent-grid orders and every error reported; round-off-saturated or zero errors make the fit indeterminate. Use dt=0.5*dx^(5/3), clipping the last step to t=1, for the order vote. The separate CFL=0.5 2x vote compares G with unlimited at the SAME N and final time (no cross-resolution ratio); the 2x L-inf inequality MUST hold at EVERY N=50,100,200,400,800 at t=1; any failing N fails this vote. This supplies the previously missing N/time specification; it does not narrow rev 9b's same-resolution rule. Unlimited means G blend/A flux limiter OFF via B2's PROPOSED debug_disable_flux_positivity while limiter:true keeps repair enabled; require observed clamp count=0, interior AND ghost, and no NaNs for a valid reference, otherwise void that case's comparison. Hu's advected-variable interpretation and paper dt/p.11 verification remain open as in rev 9b.

## Literature

- Hu, Adams & Shu 2013 JCP 242 (eq 12).
- Wong et al. 2021 JCP 444 (Alg. 4).
- Clayton et al. 2026 Comput. Fluids 317 (eq 5.1).
- Satoh 2003 (eq 9-10, 41).
- Bannon 2002 (eq 6.1-6.4).
- Li & Chen 2019 (eq 10, 14).
- Subbareddy et al. 2017 JCP 348 (eq 9).
- Larrouturou 1991.

Full notes are in the team's reading notes. This filled section supersedes the empty-Literature ADD in the 11:51 post.

## rev 9b: implementation contracts and independent oracles

All Snapy source references in this revision are to 5eeb9b6761ae484a98b3993aae18314fc58cf862, checked against that commit; Kintera references are to 4dc613d04f24621b3119d343c5c7c9b93628895b (v2.5.15), retrieved and inspected at that commit. These are study specifications, not implementation or numerical results.

### B1: existing limiter versus proposed G

- Existing A mechanics: theta is per (cell, species), not one scalar per face. Only conserved species partial-density rows ICY.. are passed to the limiter; IDN (dry mass), IVX-IVZ and IPR are not scaled by theta (src/hydro/hydro_forward.cpp:335-342,386). Independently reconstruct theta_s = min(1, max(rho_s,0)*V*(1-4096*eps)/max(dt*out_s,1e-300)), or 1 when out_s=0, with out_s the sum of outgoing area-weighted species fluxes; then F_on,s = theta_s,donor*F_hi,s (src/hydro/flux_positivity.cpp:39-73,83-103). This is the existing-code oracle, not a whole-vector LF blend. Momentum/energy instead lose the withheld species mass times donor velocity/enthalpy when hspec exists (src/hydro/flux_positivity.cpp:127-136; src/hydro/hydro_forward.cpp:380-386).
- Existing settling: fsed1 is the species flux increment from sedimentation before the limiter (src/hydro/hydro_forward.cpp:167-172). The net species flux, including settling, selects theta's donor. The carry splits advective and settling parts, using each part's own sign for its energy/momentum donor but the net-flux share for both; the final net species flux is scaled (src/hydro/flux_positivity.cpp:127-142; src/hydro/hydro_forward.cpp:382-386). fsed1 is a saved diagnostic input to carry, not a separately scaled conserved flux.
- Disagreement B1: the bot's assumed existing common-face theta/F_lo blend is absent; the inspected limiter is per-species donor scaling plus carry, so its oracle above must not be replaced by the prospective G oracle (src/hydro/flux_positivity.cpp:63-103,127-142).
- PROPOSED G flux specification: implement a new first-order LF/Rusanov low-order arm, not a relabeling of existing HLLC or A. For z=1 Cartesian tests, use piecewise-constant primitive states and U = (rho_d, rho*v, E, rho_s), E = sum_all rho_n*(R/M_n)*(u0_R,n+cv_R,n*(T-Tref)) + rho*|v|^2/2, dry included; u0_R is the original YAML value at Tref under the O3 reference contract (Kintera 4dc613d src/thermo/thermo_y.cpp:63-67). The advective physical flux has species/dry mass rho_n*v_normal, momentum rho*v*v_normal+p*normal, and energy (E+p)*v_normal. Define F_lo = (F(U_L)+F(U_R))/2 - alpha_d*(U_R-U_L)/2, with a single direction-wide alpha_d = max_cells(|v_d|+a) (including boundary donor states), a = sqrt(gamma_mix*p/rho), gamma_mix = sum(c_n*cp_R,n)/sum(c_n*cv_R,n), c_n=rho_n/M_n; gases have cp_R=cv_R+1, clouds cp_R=cv_R. This reproduces the z=1 mixture sound-speed construction, not a fixed dry gamma (src/eos/moist_mixture.cpp:221-247; W->A and WA->L dispatch at src/eos/moist_mixture.cpp:78-87). Existing HLLC receives that EOS gamma/sound speed and uses pressure-corrected L/R speeds, not this LF alpha (src/riemann/hllc.cpp:42-53; src/riemann/hllc_impl.h:34-54).
- PROPOSED G settling and composition: add a first-order upwind settling contribution to F_lo, with cloud mass, momentum and energy transported together using donor velocity and O3 h_c; F_hi is the existing reconstructed-state Riemann flux before positivity modification (src/hydro/hydro_forward.cpp:153-158), including settling before blending. For x1 use the conservative proposed bound alpha_1 = max_cells(|v_1|+a) + max_cells,species(|v_settle,s|) in LF and the step restriction, and verify positivity of this combined low-order update before allowing a G gate to pass. Blend all conserved hydro rows IDN, IVX-IVZ, IPR and ICY.. with ONE theta_f in [0,1] per face, shared across rows and neighboring blocks. G replaces both A's species scaling and ideal-moist carry; it does not run on top of them. The existing split to replace is src/hydro/hydro_forward.cpp:380-386 and src/hydro/flux_positivity.cpp:138-148. Separate scalar tracers need an explicit follow-up transport contract; they are not silently counted as hydro rows.
- PROPOSED G admissibility/open question: choose face coefficients to preserve nonnegative dry/species densities and admissible internal energy in both adjacent updated cells at each stage. The coefficient-construction algorithm and a positivity-safe combined settling/EOS timestep bound remain to be established; neither is supplied by the existing per-species code (src/hydro/flux_positivity.cpp:63-73). G is ineligible for DECIDE until these are implemented and validated, including the stated no-clamp tests; a mixture/settling positivity proof must not be inferred from the existing HLLC speeds.
- PROPOSED G oracle: on EVERY face and EVERY limited row, independently reconstruct F_lo in Python from primitive states, YAML thermodynamics and settling speeds (no calls to the implementation's flux/EOS conversion), and check F_on,r = theta_f*F_hi,r + (1-theta_f)*F_lo,r with the SAME theta_f for all rows. Log all row residuals, face coefficients and shared-face equality. Relative tolerance PROPOSED 1e-12, normalized by max(|F_on,r|, |theta_f*F_hi,r|, |(1-theta_f)*F_lo,r|); require exact zero if this scale is zero. Also check the existing A oracle above independently. A-only dry-flux equality, A-only carry residuals and A-only Gate 4 bitwise checks are not G constraints; conservation/positivity and full regression gates apply to both.

### B2: unlimited reference and shared diagnostics

- limiter: false switches off conserved cell repair AND primitive repair, as well as the flux limiter (src/eos/equation_of_state.cpp:213,358; src/hydro/hydro_forward.cpp:335). The inspected option is a single EOS limiter bool (src/eos/equation_of_state.hpp:49-54); the inspected flux path has no independent cut switch. PROPOSED required code change: add debug_disable_flux_positivity, leaving limiter: true for cell repair and bypassing only A's flux cut/carry or G's blend. It must not disable sedimentation. This is the unlimited-reference arm for both numerical comparisons and on/off flux tests, and is part of the arm implementation plan, not an existing option.
- Every TESTS (2) run, including the unlimited reference, must log per-cell NaNs BEFORE repair, clamp counters/masks and mass changes at every step/stage (interior and ghost separately). A NaN OR any clamp in the reference voids that case's accuracy comparison even if repair removes it; report every clamp in the reference. DECIDE's no-clamp clause requires zero actual clamps in G, including ghost cells at every step/stage, not just zero marked blocks; count interior and ghost clamps separately but either violates no-clamp. Existing code replaces NaNs and reports thresholded species marks, not the required census (src/eos/equation_of_state.cpp:217-219,348-352,359-364,384-385); meshblock exposes two bools and uses them in redos (src/mesh/meshblock.cpp:1070-1077,1129-1144). Add the counters before mutation, not after the data have been repaired.

### O3, mutation tests and thermodynamic tolerances (items 1-4, 11)

- Evaluate O3 at EACH cell's own YAML composition, p, rho and nonzero kinetic energy, including dry, vapor and cloud. The printed card enthalpies below are KE=0 illustrations only, not shared expected values. Add a cell with nonzero vapor u0_R (PROPOSED -1000 K) as well as nonzero velocity; evaluate h_n independently from the displayed formulas. The existing ideal-moist hook includes both u0 and metric KE (src/eos/ideal_moist.cpp:265-279); the new moist-mixture A hook must satisfy the same independent checks, not inherit the default undefined hook (src/eos/equation_of_state.hpp:123-126; src/hydro/hydro_forward.cpp:380-384).
- Run C, dropped inv_mu, cloud + R_c*T, cloud cv_R=3.5 and cloud-as-vapor as five separate mutations injected into species_enthalpy (the hook consumed at src/hydro/hydro_forward.cpp:364,382-384), not just wrong Python answers. For EACH mutation both the affected carry cells and O3 must go red against unchanged independent references; restore the hook between mutations. The old Gate 1 control run is superseded by this production-path mutation requirement. Gas/cloud coefficients and units being mutated are visible in src/eos/ideal_moist.cpp:265-279; implementing the missing moist-mixture hook is arm A work.
- Print use_nasa9_cp and use_h2_cp in EVERY run log; a cell fails if either is on. Verified at Kintera src/thermo/eval_uhs.cpp:288-300: the linear uref_R + T*cref_R result is overridden by NASA-9/H2 paths at :289-295 and :296-300. This precondition applies to every arm and every mutation, not only the superseded Gate 1b.
- A-only face energy/momentum residuals test carry mechanics. The current energy reference uses the code's own W->E and adds vapor RT (tests/test_flux_positivity_carry.cpp:94-115), so pair each such cell with YAML-only O3 on the SAME state; that residual alone cannot certify enthalpy. W->E's conversion is src/eos/moist_mixture.cpp:184-193; its kg/m3 argument issue remains a separate fix. G instead uses the independent whole-vector blend oracle and column conservation.
- h_n at 1e-12 relative remains primary. Retain the conservative PROPOSED T tolerance 7e-14 relative and 1e-12 K absolute pending sign-off, but supersede the old 12.7x justification: with the corrected cloud h_c=(R/M_c)*(-3430+9*(T-300)), |d ln |h_c|/d ln T|=|9*T/(-3430+9*(T-300))|=1.0780549562404285 at T=353.34734862459709 K. The old 12.7267535627529 used an unshifted YAML offset and is invalid for this card. Recompute conditioning at each actual state including KE; near-zero h uses the absolute enthalpy tolerance. No tolerance is silently relaxed.

### P, CFL and non-vacuous coverage (items 5-6, 9-10)

- P requires positivity hits > 0 in addition to zero marked cells; otherwise the run is vacuous and cannot establish a limiter guarantee. Existing positivity_hits counts (cell,species) theta<1, not limited faces (src/hydro/hydro_forward.cpp:343-346); record BOTH that count and a new face count for A/G, with nonzero G face count required by Hu. Instrument clamp attribution before/after transport and before/after kinetics/forcing and report causes separately; external forcing is applied after divergence (src/hydro/hydro_forward.cpp:403-421). No assertion about an uninspected external kinetics driver substitutes for this instrumentation.
- For each step require interior mass change - net inward boundary mass flux integrated over the step - recorded interior clamp mass = 0 to PROPOSED 1e-12 relative, with scale max(initial interior mass, absolute budget terms). Keep ghost clamp mass separate. This formula applies to the transport/source-free budget; if forcing/kinetics adds physical mass, also subtract its measured source increment in the full-step closure and report both budgets. Disagreement 5: an unqualified full-step source-free identity is not valid when forcing adds mass, because forcing modifies du after transport (src/hydro/hydro_forward.cpp:415-422). Zero redos alone cannot settle the new non-vacuous clamp/budget gate (src/mesh/meshblock.cpp:1129-1144).
- Snapy's explicit hydro timestep is min_d,cells dx_d/(|v_d|+a), then limited by diffusion; the block/global minimum is multiplied by cfl and 2^(-current_redo) (src/hydro/hydro.cpp:125-133,190-212; src/mesh/meshblock.cpp:507-525). The implicit-correction branch has different denominators/shear bounds (src/hydro/hydro.cpp:145-187). For explicit uniform 2D define tau_x=max(|v_x|+a)/dx and tau_y=max(|v_y|+a)/dy: absent other restrictions dt_snap=cfl/max(tau_x,tau_y), whereas Hu eq. 29 uses dt=CFL_Hu/(tau_x+tau_y). Log actual dt, both rates and CFL_Hu=dt*(tau_x+tau_y) at every stage of the 0.9/0.5/0.3 ladder; input cfl=0.5 is NOT necessarily Hu CFL<=0.5 (equal rates give CFL_Hu=1). PROPOSED additional capped control dt<=0.5/(tau_x+tau_y), including the G settling speed restriction above, is required for a claim at Hu CFL<=0.5. Keep and report the uncapped cfl 0.9 outcome either way. Paper equation attribution remains per oracle review pending the PDF.
- All TESTS (2) runs use the B2 per-cell clamp instrumentation, not just CRM. G's positivity/no-clamp gate and both arms' Larrouturou Y vote are independent of the accuracy norms. A fallback under DECIDE must report its own failures; a failed A interval vote is not silently promoted to a passing maximum-principle guarantee.

P clarification (rev 9c): retain nonzero positivity hits, causal clamp attribution and mass closure; zero marked cells alone is not a pass. For DECIDE's no-clamp claim, require zero actual interior AND ghost clamps, including round-off clamps, in all TESTS (2) and qualifying P runs. The CRM CFL=0.9 run is still reported whether it passes or fails. Historical P wording below remains superseded.

Second carry card (PROPOSED): test_flux_positivity_carry_vapor_u0.yaml, following the existing test_flux_positivity_carry.yaml name (tests/test_flux_positivity_carry.cpp:26-27 at 5eeb9b6); rev 9b specified the card's state but no filename. Copy the base thermodynamic card and set vapor u0_R=-1000 K at the card's Tref=300 K (not at T=0), retaining the cloud offset and nonzero velocities. The carry binary loads one thermodynamic base card per process (source comment/kCard at :26-27, YAML::LoadFile at :41); it writes temporary on/off variants at :44-51, so this does NOT mean one YAML read per process. PROPOSED harness change: choose the base-card path once at process startup and invoke a fresh process for each card, never swap species tables within a process. Existing one-step calls are at :68-69 and the nonzero (vx,vy)=(2,3) comparison at :175-176.

| Additional card/run | Arm | Device | N | Step count | Gates/oracles |
| --- | --- | --- | --- | --- | --- |
| test_flux_positivity_carry_vapor_u0.yaml (PROPOSED) | A plus B2 unlimited reference; ideal-moist and moist-mixture, lmars/hllc | CPU and CUDA, fp64, separate processes per card/device | 6x1x1, 2 ghosts (base-card geometry; tests/test_flux_positivity_carry.yaml:27-30) | One hydro forward with dt=1 per arm, as source :35,69; proposed new-card run | Gate 2 CPU / Gate 3 CUDA; own-state O3, I1/I2 and A carry energy/momentum residual; five hook mutations must turn affected carry/O3 checks red |

The second card is an additional A thermodynamic-regression row beyond the 24 primary device/configuration cells, not an extra G positivity-at-CFL<=0.5 claim: its dt=1 drain test follows the carry fixture. All new card/harness work is proposed implementation work, not added files in this documentation commit.

### Rev 9b responses

The following mapping covers every requested item; line numbers in this historical table refer to rev 9b, except the explicitly updated item 7 which points to rev 9c. PROPOSED items remain open for the restarted sign-off, including the new G algorithm/settling contract and Hu paper verification.

| Item | PLAN.md lines | Response |
| --- | --- | --- |
| B1 | 129-135,251,316-317,349 | Existing per-species oracle; proposed G, settling and wave speed; disagreement at line 131. A-only gates also scoped below. |
| B2 | 106,139-140 | Independent flux-disable knob required; repair and NaN/clamp logging retained; Hu timing pending paper. |
| 1 | 144,290 | Own-state O3, nonzero KE and vapor reference energy. |
| 2 | 145 | Five hook mutations; carry and O3 must fail; old Gate 1 superseded. |
| 3 | 146 | Both cp flags printed/off; Kintera lines checked. |
| 4 | 147,316-317 | A carry energy residual paired with O3 on the same cells. |
| 5 | 152-153,374 | Nonzero hits, attribution and mass closure; disagreement: physical mass sources must be included in full-step budgets (src/hydro/hydro_forward.cpp:415-422). |
| 6 | 154 | Actual Snapy-to-Hu CFL conversion and capped control. |
| 7 | rev 9c L123-L124 | Fixed in rev 9c: interface positivity only, initial trace density; L1 order moved to a separate smooth trace card; contact diagnostics retained. |
| 8 | 108,155 | Both A and G receive the Y-interval vote; unlimited excursion alongside each. |
| 9 | 106,152 | Hu offset variable is PROPOSED pending PDF verification; nonzero limited faces required. |
| 10 | 140,155 | Per-cell clamp census in every literature-test run. |
| 11 | 148,310 | Primary enthalpy tolerance retained; old 12.7267535627529 amplification superseded by rev 9d (1.0780549562404285); conservative proposed T budget retained. |
| attribution | 6,9,106-108 | Rev 9a review credit corrected to Sihe Chen; prior approve superseded; genuine Xi 12:58 attribution retained. |

## Rev 9c responses

Items 1 and 2 credit Sihe Chen (Slack 1790887060.769699); 3a-3h address the independent rev 9a review as supplied in this brief (the dungeon2 review file was not locally inspected). bot-9c-1..3 credit the coordinating bot's 9c review (Slack 1790887177.825939). Current line numbers below refer to rev 9c; explicit rev 9b numbers identify prior closures.

| Item | Status | PLAN.md lines and response |
| --- | --- | --- |
| 1 | fixed | L123-L124: interface positivity only, initial trace density; separate concrete smooth L1 order test. Rev 9b item 7 corrected. |
| 2 | fixed | L125-L126: CFL 0.5 proposal versus paper 0.75, paper/study Y_1 mappings, both arms; voting interval superseded by bot-9c-1. |
| 3a | fixed | L294: I1 retargeted from dropped O1 to surviving O3, paired carry check. |
| 3b | fixed | L295: independent inline S_n definition. |
| 3c | fixed | L399-L400: active z=1 A/G counts and arm-specific oracles replace 18 x A/B. |
| 3d | already closed in 9b at L349 | A-only Gate 4 bitwise checks retained at L401. |
| 3e | already closed in 9b at L140,152-155,374; clarified | L160, L177: actual clamps, not marks; ghost clamps explicitly count, reference must also have zero clamps. |
| 3f | fixed | L122, L128: N=50-800, t=1, fit convention, B2 unlimited recipe and zero-clamp reference. |
| 3g | note | L15: history now logs the unlogged rev 9a L338 cross-reference repair; historical grid/decision fence unchanged, and its rev 9b criteria link resolves to the current rev 9c criteria. |
| 3h | note | L16: '7 numerics papers' versus five numerics citations remains an unresolved count discrepancy, left as written because the 12:58 block is verbatim. |
| bot-9c-1 | fixed | L125-L126: nontrivial initial-range vote, both arms; pure-species overshoot comparison reported, not voting. |
| bot-9c-2 | fixed | L122: both Hu checks vote; same-resolution CFL=0.5 comparison preserved. |
| bot-9c-3 | fixed | L179-L183: named second card, run row and separate-process requirement; source confirms one base card, not one YAML read. |

No requested item was rejected as a code contradiction. The retained proposed study gas pair differs from Larrouturou's paper pair; this is explicitly an adapted reported comparison. The frozen 12:58 block's Wong 'floor' wording and combined tests list remain historical verbatim text; the current executable criteria are the rev 9c criteria above.

## critique -> change, scope

Superseded by rev 9: the historical non-ideal branch-(a) scope, 36-cell count, G-AB and A/B eligibility below are not active gates. Rev 9 uses 24 z=1 cells and the 12:58 A-versus-G DECIDE.

```text
Scope: branch (a) only. A test-only kintera option registers virial functions in the CPU and device func2 tables, so all 24 moist-mixture cells run, including the 12 z!=1 cells. Branch (b) and its TORCH_CHECK are removed. Drafting only; nothing run.

Critique -> change (Tianhao's bot review of rev 3)
1. B had no device-supported ddC contract -> part 2 lists what kintera main has today (CPU and CUDA, file:line), what the test-only registration adds, and a new eval_intEng_R_ddC contract: signature, units, return value. The O1 derivatives are analytic: dz/dc = B, du/dc = -a. B is wired through the production species_enthalpy path. New z!=1 CPU and CUDA carry cells compare against O1 and O2 (parts 4-5).
2. Invariants were implicit -> part 2 states them as equations with tolerances:
- I1 separability: sum over all species of c_n M_n h_n = U + p + rho KE per volume.
- I2 locality: h_n depends only on the cell's own state.
- I3 Maxwell consistency: (dU/dV)_T,N = T (dp/dT)_V,N - p.
- I4 Euler/FD: hbar_n = (dH/dN_n)_T,p (O2).
3. Dry gas was left out -> dry is now in every reduction: I1, G-AB, the dry mass flux and counts, the column totals (IDN and ICY rows added) and O3 (h_d added). Part 2 lists the rev 3 omissions.
4. No inconsistent-model test -> new control X: u without the -a c term while z keeps -a/(RT). It fails I3 by 6.906e5 Pa against a 7.9e-3 Pa tolerance (8.7e7 x).
5. A must stay supported -> no TORCH_CHECK and no behaviour change. New test A-NI: formulation A on the z!=1 model with a non-empty intEng extra must satisfy I1 exactly and pass the carry cells. The decision rule picks A or B on the gates and does not reject A.

Kept from rev 3 (chen sihe, agreed)
- Gate 0 records the six moist-mixture limited-face counts on main and does not abort on a mismatch. It requires all six red and the ideal-moist arms green. (Rev 5: main is 5eeb9b6.)
- Gate 4 is the full ctest plus the python limiter tests. (Rev 5: plus the two bitwise checks against 5eeb9b6.)
- G-AB (A = B bitwise-to-4-ulp where z=1).
- The dropped inv_mu control (fails at rel about 0.97-0.98, not 1e-12).
- The W->E kg/m3 bug (moist_mixture.cpp:184-185) stays its own fix, never folded into A or B.

Kept from rev 2 (Tianhao)
- A is a hypothesis, not the established partial-molar enthalpy.
- The oracles are standalone formulas with inputs.
- Every cell is listed; CPU-only or skip = gap.
- Each observable has abs + rel tolerances.
- The controls come with expected failure sizes.
- O3 uses YAML constants only, never W->E.

Cell count: 36 = 24 moist-mixture gate cells (6 cases x {z=1, z!=1} x {CPU, CUDA}) + 12 ideal-moist controls. Under (a) all 36 are runnable, and none is uncovered by design.
```

## kintera contract, invariants, dry gas

Superseded by rev 9: the test-only virial option, new eval_intEng_R_ddC API/kintera PR, B, I3, I4 and G-AB below are dropped from this study. I1, I2, dry inclusion and the cp-override precondition remain in the z=1 scope.

```text
kintera main (4dc613d) today
- func2 is f(T, c_n) per species; call_func2 adds to a preset value (CPU: call_func2_cpu, src/utils/utils_dispatch.cpp:47-69, REGISTER_ALL_CPU_DISPATCH at :110; CUDA: call_func2_cuda, src/utils/utils_dispatch.cu:42-65, REGISTER_CUDA_DISPATCH at :102).
- The CPU table is empty (func_table.cpp:30-32). The CUDA table is {nullptr} (func_table.cu:39-43). Names are looked up via get_device_func2 (user_funcs.cu:54).
- eval_czh (eval_uhs.cpp:223-243) and eval_czh_ddC (:245-264) exist on CPU and CUDA.
- eval_intEng_R with intEng_R_extra (:266-302) exists. Its T-derivative is taken via the name + "_ddT" (eval_cv_R :169-174; thermo_dispatch.cpp:103-110, .cu:98-103).
- There is no c-derivative of intEng_R_extra. No eval_intEng_R_ddC exists.

The test-only option adds (one set per species, for dry gas and vapour ONLY; no cloud extra is registered, so every cloud czh, czh_ddC and intEng extra stays unset; constants compiled in; they return increments because call_func2 adds)
- z_virial_<sp>(T, c) = B(T) c, onto czh = 1
- z_virial_<sp>_ddC(T, c) = B(T), with B = b - a/(RT) in m3/mol
- u_virial_<sp>(T, c) = -a c / R, in K
- u_virial_<sp>_ddT = 0, required: eval_cv_R appends _ddT, and a missing non-empty name throws "not registered" (user_funcs.hpp:23-39 CPU, user_funcs.cu:54-78 device)
- u_virial_<sp>_ddC = -a / R, in K m3/mol
- The CUDA table is indexed id+1 behind a leading nullptr, so it must list the same functions in the same order as func2_names.

New production API (kintera, general; test functions stay test-only)
- eval_intEng_R_ddC(temp, conc, op) -> tensor shaped like conc.
- Entry n = d(u_n/R)/dc_n in K m3/mol at fixed T. It is 0 where no extra is set.
- It calls call_func2 with the intEng_R_extra names + "_ddC", on CPU and CUDA, and mirrors eval_czh_ddC.
- Inputs: temp in K; conc in mol/m3 (conc = V * inv_mu, moist_mixture.cpp:223 and :236 on 5eeb9b6).
- Snapy B uses eval_czh_ddC and eval_intEng_R_ddC. It computes all species (dry included), divides each molar term by its own M_n (h *= Rgas*inv_mu, as W->E does at moist_mixture.cpp:184 on 5eeb9b6), and returns rows 1..ny.

Invariants (|x-y| <= abs + rel*max)
- I1: sum over dry, vapour and cloud of c_n M_n h_n = U + p + rho KE. Tolerance 1e-6 J/m3 + 1e-12. Evaluate at each surviving O3 cell including dry and nonzero KE, using independent YAML U=sum_n rho_n*(R/M_n)*(u0_R,n+cv_R,n*(T-Tref)) and p=R*T*sum_gas rho_n/M_n; use YAML u0_R at Tref as in O3 (Kintera 4dc613d src/thermo/thermo_y.cpp:63-67); report the measured residual per cell, not the dropped O1 number. Pair A carry residuals with O3 on the same state as specified in rev 9b item 4.
- I2: h_n in cell i is unchanged, bitwise, when the neighbours change. h - KE is unchanged when v changes, within 4 eps S_n, eps=2^-52, S_n=(R/M_n)*(|u0_R,n-cv_R,n*Tref|+cv_R,n*T+z_n*T)+|KE|, where z_n=1 for gas and 0 for cloud in this z=1 study; this definition is independent of the dropped G-AB block.
- I3: (dU/dV)_T,N = T (dp/dT)_V,N - p, by Richardson finite difference on the kintera VT->U and VT->P path. Tolerance 1e-3 Pa + 1e-8. At O1: 690600.0 = 690600.0, residual 2.2e-5 Pa.
- I4: B = O2 within 1e-6 J/mol + 1e-8.

Dry gas left out in rev 3, now fixed
- The sum identity: a new test computes h_dry with the A formula by hand (842a116's nasa9 test did this; 5eeb9b6 has no such test). Under B it must use B's dry value.
- G-AB covered only the vapour and the cloud; the dry row is added.
- The column totals covered only IPR and IVX-IVZ; IDN and every ICY row are added, exact to the same tolerance.
- The counts (arm A only): the dry mass flux on vs off must be exactly equal, on every face, CPU and CUDA (src/hydro/hydro_forward.cpp:335-339,386 on 5eeb9b6). G uses the rev 9b whole-vector oracle instead.
- O3 had no h_d; it is added (part 3).
- The dropped inv_mu control for dry was listed only for the sum; it is now also a per-species gate.
```

## oracles and controls

Superseded by rev 9: B, O1, O2, X and the non-ideal A comparisons below are historical only. O3 remains YAML-only on z=1 for every species; C, dropped inv_mu and condensate controls are retained in that scope.

```text
B (J/mol, then divide by M_n)
- zeta = z + c z'; vbar_n = zeta_n / sum c zeta; phi_n = c_n (u_n' + R T z_n').
- Gas: h_n = u_n + z_n R T + phi_n - vbar_n sum c phi.
- Cloud: h_c = u_c.
- A = u_n + z_n R T.
- Assumptions: p = R T sum c z(T, c_n); clouds have zero volume; fixed T and p; YAML u0_R is molar internal energy/R at Tref, not the T=0 intercept; u_n=(R/M_n)*(u0_R+cv_R*(T-Tref)) in the ideal scope. R = 8.31446 J/(mol*K) (Kintera 4dc613d src/constants.h:6); Tref comes from reference-state.Tref (src/thermo/thermo_options.cpp:47-48), 300 K for this card. Historical O1/non-ideal numbers below are inactive, not recalculated results.

O1 (closed form, standalone)
- z = 1 + B c, B = b - a/(RT); u = u0 + cv T - a c.
- Dry: a 0.137, b 3.87e-5, cv 2.5R, u0 0, M 28.97e-3.
- Vapour: a 0.5536, b 3.05e-5, cv 3.5R, u0 -4.4e4 J/mol, M 18.015e-3.
- Scope: virial z and u for dry gas and vapour ONLY. Clouds get no extra (z, u extras unset), so h_c = u_c is the ideal value in every z!=1 cell.
- Dense state: T 353 K, c 1000/1000 mol/m3. Rechecked, same values as rev 3:
  - dry 10646.79 J/mol = 367511.0 J/kg
  - vapour -32344.85 J/mol = -1795439.8 J/kg
- Card state (rho 1, y 0.97/0.01, p 1e5, z!=1): T 353.440 K, B dry 354845.6 J/kg, B vapour -1708124.1 J/kg.
- A-B at the card state: dry +2.70 J/kg (7.6e-6), vapour -262.29 J/kg (1.5e-4). These are the z!=1 carry cells' discriminating sizes.

O2 (finite difference, using only the model's p and U)
- Fix p* = p(V=1). Newton-solve for V at N'; H = U + p*V.
- Central difference in N_n, step 1e-5 N_n, plus Richardson; divide by M_n.
- Matches O1 to 1.1e-10.

O3 (YAML only, z=1)
- Reference contract: use original YAML u0_R and Tref, not the already shifted runtime uref_R. With cp overrides off, u_n=(R/M_n)*(u0_R+cv_R*(T-Tref)); R=8.31446 J/(mol*K), Tref=300 K for tests/test_flux_positivity_carry.yaml:4-6 at Snapy 5eeb9b6. Kintera 4dc613d src/thermo/thermo_options.cpp:47-48 reads Tref from the card (default 300, src/thermo/thermo.hpp:77); src/thermo/thermo_y.cpp:63-67 subtracts cv_R*Tref once and :82 marks offset_zero; src/thermo/eval_uhs.cpp:285-288 adds cv_R*T to that shifted intercept; src/constants.h:6 fixes R. Dry/vapor absent u0_R defaults to 0 (src/species.cpp:169-172); cloud YAML u0_R=-3430 K is at Tref.
- Both Gate 0 EOS paths use this shift: Snapy 5eeb9b6 src/eos/ideal_moist.cpp:18-20 constructs ThermoY and :47-50 stores its shifted uref_R; src/eos/moist_mixture.cpp:18-20 also constructs ThermoY and :184-185 evaluates its energy. These source references are checked, not a proposed change to either implementation.
- T = p / (R sum_gas rho y / M)
- h_d = (R/M_d)(u0_R,d + cv_R,d (T - Tref) + T) + KE (u0_R,d=0 in this card)
- h_v = (R/M_v)(u0_R,v + cv_R,v (T - Tref) + T) + KE
- h_c = (R/M_c)(u0_R,c + cv_R,c (T - Tref)) + KE
- Card: Tref=300 K; dry cv_R=2.5, u0_R=0; vapor cv_R=3.5, u0_R=0; cloud cv_R=9.0, u0_R=-3430 at Tref. Gate 0 measured M_d=0.02897 and M_v=M_c=0.018015 kg/mol. The uniform helper uses rho=1, p=1e5, Y_d/Y_v/Y_c=0.97/0.01/0.02 (Snapy 5eeb9b6 tests/test_flux_positivity_carry.cpp:59-65).
- Values PROPOSED for signer confirmation (recomputed YAML-only, KE=0 illustration): T=353.34734862459709 K; h_d=139688.58774105753, h_v=249255.21971155726, h_c=-1361454.8006545985 J/kg. Supersedes the old printed T=353.347 K, h_d=3.54940e5, h_v=7.33861e5, h_c=-1.15325e5. Actual O3 gates still use each cell's own state, nonzero KE and the separate nonzero-vapor-u0_R card; do not compare every cell against these illustration values.

Controls, each must miss by >= 1e3 x its tolerance
Rev 9d: the old O3 C=0.51 and condensate relative miss sizes below used the wrong reference and are superseded, not active expected sizes; remeasure each hook mutation against corrected O3. Inactive O1/non-ideal values remain historical.
- A on O1 (dense): rel 5.0e-2 dry, 1.7e-2 vapour.
- A on the card z!=1 cells: 7.6e-6 dry, 1.5e-4 vapour.
- C = (U+p)/rho_gas on O1: rel 2.26 dry, 0.74 vapour. On O3 vapour: 0.51.
- Dropped inv_mu: rel 0.971 for dry (a factor of 34.5), 0.982 for vapour and cloud (55.5).
- Condensate: cloud + R_c T 1.41; cloud with cv 3.5 7.78; cloud as a vapour 7.36. K2's p/rho_c is reported at 8.7e-4.
- X, inconsistent (u = u0 + cv T, z unchanged):
  - I3 residual -6.906e5 Pa vs 7.9e-3 Pa tolerance (8.7e7 x).
  - Its B shifts by -4607 J/kg dry (1.3e-2) and +45744 J/kg vapour (2.5e-2) against O1.
  - O2 agrees with X's own formula, so I3 is the gate that catches it.

Non-circular: any other split is h + delta with sum c delta = 0 and delta != 0. O1 and O3 pin every h_n, and O2 pins dH/dN_n.
```

## tolerances, G-AB, gates

```text
Tolerances (abs / rel)
- T vs code: PROPOSED 1e-12 K / 7e-14 (conservative budget retained in rev 9d; old 12.7x conditioning superseded; h_n remains primary)
- h_n vs O1 or O3, all species including dry: 1e-9 J/kg / 1e-12
- O2 vs O1: 1e-6 J/mol / 1e-8
- I1: 1e-6 J/m3 / 1e-12
- I3: 1e-3 Pa / 1e-8
- limited counts vs Gate 0, and the dry mass flux: exact for arm A only; G uses its own face census/oracle
- face energy residual (arm A only, paired with O3 on the same cell): 1e-9 / 1e-12
- face momentum residual (arm A only): 1e-12 / 1e-12
- column totals IDN, ICY.., IPR, IVX-IVZ: 1e-12 / 1e-12 of sum|du|
- CPU vs CUDA:
  - h_n: 1e-9 / 1e-14
  - theta: 1e-15 / 1e-13
  - species flux: 1e-15 / 1e-13
  - energy flux: 1e-9 / 1e-13
  - momentum flux: 1e-12 / 1e-13
  - counts: exact
- code M vs composition M: rel 2e-5
Superseded by rev 9: the following A-vs-B runtime gate is dropped.
- runtime: median of 5; soft gate B <= A + 15%

Superseded by rev 9: G-AB is dropped; the historical block below is retained for reference. Rev 9d: its u0_R scale would require the shifted T=0 intercept, not raw YAML u0_R; its printed bounds are inactive. Active I2 defines the corrected scale inline.
G-AB (z=1 cells only)
- |h_A - h_B| <= 4 eps S_n, eps = 2^-52, S_n = (R/M)(|u0_R| + cv_R T + z T) + KE.
- It applies to dry, vapour and cloud, including ghosts.
- With no extras registered, czh = 1 on gases and 0 on clouds, czh_ddC = 0 and extra = 0 exactly (eval_uhs.cpp:225-226, 247, 268-288), so B = A + 0. (use_nasa9_cp and use_h2_cp off; eval_uhs.cpp:289-300 overrides intEng otherwise)
- Cloud half: a cloud has z = 0 (czh = 0 on every cloud slot, which start at vapor_ids().size(), thermo_y.cpp:78), so S_n for a cloud drops the z T term, and intEng_R = uref_R + T cref_R exactly unless the NASA-9 or H2 cp path overrides that slot (eval_uhs.cpp:289-295, 296-300); an overridden slot is checked against its override, not against uref_R + T cref_R.
- Bound at the card: cloud 2.7e-9, vapour 6.5e-10 J/kg.
- It covers the 12 z=1 cells.
- If it fails, B is not the claimed reduction and is stopped.

Gates
- 0 (CPU): build current main 5eeb9b6 (v2.10.36) against kintera v2.5.15 (4dc613d, kintera main), the same for both builds so the bitwise checks reproduce. Record the six moist-mixture counts, M_n and T. The earlier 3f7ad96 counts are a prior only. Pass if all six moist-mixture cases are red (energy residual rel 1.0 at every limited face) and every ideal-moist arm is green. A count mismatch against the prior is recorded, not an abort. 842a116's after-numbers are recorded as a candidate's, not as an oracle.
Superseded by rev 9: Gate 1 below depends on dropped O1/O2/I3/X work; it is not an active gate.
- 1 (CPU, no snapy): O1/O2 are reproduced in J/mol and J/kg, including the card z!=1 values. I1 and I3 hold on O1. Every control misses by its stated size (+-10%) and by >= 1e3 x tolerance, X included.
Superseded by rev 9: Gate 1b below and its test-only kintera work are dropped; keep use_nasa9_cp and use_h2_cp off for the retained scope.
- 1b (CPU and CUDA, kintera test build): scope is dry gas and vapour ONLY; every cloud extra must read back unset (cloud czh = 0, czh_ddC = 0, intEng_R = uref_R + T cref_R exactly; kintera 4dc613d eval_uhs.cpp:225-226 sets czh = 1 only on the vapor_ids slice, dry included; cloud slots start at vapor_ids().size(), thermo_y.cpp:78). The registered functions match O1 per species: czh, czh_ddC, eval_intEng_R and eval_intEng_R_ddC to 1e-12. I3 holds on the VT->U and VT->P path. X, registered in the same build, fails I3. (use_nasa9_cp and use_h2_cp off; eval_uhs.cpp:289-300 overrides intEng otherwise)
Superseded by rev 9: Gates 2 and 3 below are restricted to the 24 z=1 cells (12 moist-mixture plus 12 ideal-moist controls); compare A versus G under the 12:58 plan, not A versus B.
- 2 (CPU): 12 z=1 configurations (6 moist-mixture plus 6 ideal-moist) run for each of A and G, each with its B2 reference. A meets O3/I1/I2 and per-species carry residuals paired with O3; its five species_enthalpy mutations go red. G meets the independently recomputed whole-vector blend oracle, row conservation and positivity/no-clamp contract, not A carry residuals. The second vapor-u0 card is an additional A regression row. No B, O1/O2, A-NI or G-AB requirement remains in this gate.
- 3 (CUDA, fp64): the same 12 z=1 configurations per A/G arm and the additional A vapor-u0 row meet their Gate 2 arm-specific checks plus applicable CPU/CUDA tolerances. This is 24 primary device/configuration cells across Gates 2 and 3, each exercised for both arms; extra thermodynamic/literature rows are reported separately. A skip is a gap and fails the gate; no B, O1/O2, A-NI or G-AB requirement.
- 4: full ctest (+ python limiter tests) shows no new failures vs 5eeb9b6 except moist_mixture_withholds_no_energy_or_momentum_yet, replaced by expect_carried(off, on) as its comment directs, plus two arm-A-only bitwise checks, both against 5eeb9b6 (neither constrains G): (1) ideal-moist fluxes are bitwise equal; (2) formulation A with no extras is bitwise equal to 5eeb9b6 in every flux row on every unlimited face (a face where every species' donor theta == 1, i.e. share == 0; flux_positivity.cpp:127-136 on 5eeb9b6), and in the species rows on limited faces, for one forward pass. 842a116 is a candidate only, never the oracle. No rebase comparison. Column totals hold. Runtime recorded.
Superseded by rev 9: coverage is 24 z=1 cells, with a skip still a gap; the old 36-cell requirement below is inactive.
- Coverage: all 36 cells ran on their device and passed.
```

## positivity: face flux limiter vs cell clamp (new in rev 5)

Superseded by rev 9 where the run/decision wording differs: retain P's instrumentation, interior/ghost budgets, CPU table and CUDA deciding evidence, but run the 12:58 CFL ladder (0.9 / 0.5 / 0.3) on both devices and use its DECIDE for A versus G.

```text
P. Where is species positivity guaranteed?
- Two places today: the face flux limiter (limits the species flux so no donor goes negative) and the cell repair in apply_conserved_limiter_ (equation_of_state.cpp:253-354 on 5eeb9b6, in step order: parent borrow, a parented cloud borrows its deficit from its parent vapour in the same cell (:269-285); parentless fix_vapor + clamp_min_(0) on the clouds (:302-328); vapour pass via columnar fix_vapor with TORCH_CHECK (:330-344); marks (:348-353)).
- Question: which one is the guarantee? If the face limiter guarantees it, the cell clamp should fire only at round-off; if it does not, the clamp is the real guarantee and its side effects are part of the scheme.

What the cell clamp does (to be measured, per clamped cell and per column)
- Mass: In the interior, the parent borrow and fix_vapor conserve mass; mass is added only by clamp_min_(0) on a parentless cloud with a negative column total (delta = -min(rho_c,0)). The clamp at equation_of_state.cpp:327 (5eeb9b6) acts on all of cons (ghost cells and every cloud), while the parentless fix_vapor at :321 acts on the interior only, so parentless clouds in ghost cells gain mass without passing through fix_vapor first. Record the added mass (interior and ghost separately) and the mass moved between cells.
- Energy: the clamp changes species without changing IEN, so the implied T and p shift. Record the change in total energy vs the energy the added mass would carry (h_c or h_v at the cell state).
- Momentum: IVX-IVZ are untouched, so the velocity rho v / rho changes when rho changes. Record the momentum and KE change.
- Reporting: the species repair sets limiter_marks_[0] only when a change exceeds positivity_roundoff * rho (#256); below that it is silent. limiter_marks_ is a 2-element bool per block (equation_of_state.cpp:385, torch::zeros({2}, ...) on 5eeb9b6), not a per-cell count. Add a per-cell counter / mask dump in the study build to record clamped cells, cells exceeding the marking threshold ("marked cells"), and the largest |delta|/rho per step. The marks see only the interior (:255 cons.index(interior), :349), so they are blind to ghost-cell mass added at :327; instrument the clamp before/after and record added mass for interior and ghost separately, as above. The marks drive redos (meshblock.cpp:1070 limiter_patch_hit() exposes the first flag; check_redo reads limiter_hits() at :1131-1144, and mesh.cpp:476-493 does so mesh-wide). Superseded by rev 9b for the stronger P gate (hits, actual clamps and budgets required): the historical inference was that #263's CPU run_hydro result, 4000 cycles / 0 redos, already settles the CPU marked-cell decision; the new deciding evidence in P is CUDA.

Reproducer
- 2D moist Jupiter CRM (H2O + NH3, 100x100, ideal-moist, limiter: true, rk3, cfl 0.9) from #263; with the external driver (path+sha recorded) species repairs above round-off occur nearly every step; with run_hydro, 0 redos.
- Context from #263 (closed, not a snapy source defect): with the external driver applying kinetics to a stale hydro_w, ~30k cells start each step with negative species (min/rho -8.8e-13) and every redo is a species clamp; with hydro_w refreshed first, 4000 cycles run with 0 redos and the negatives are denormal (min/rho -1.1e-303); #263's run_hydro row (#257 order) is 4000 cycles, 0 redos, and "-" for negative species at step start, so the denormal negatives were measured only with the refreshed external driver. Measured on the 5eeb9b6 tree (fa136b5), CPU.
- Whether a species-only clamp should request a redo (#226's choice) is decided here, not in #263.

Pass for P: a table per run of clamped cells, marked cells from the study-build per-cell counter / mask dump (not the 2-element block bool), max |delta|/rho, and the mass, energy and momentum change per step, CPU and CUDA; record interior and ghost added mass separately because the marks observe only the interior. Superseded by rev 9b: the following zero-marks-only rule and CPU-settled conclusion are historical; require hits > 0, attributed per-cell clamps and mass closure under the new P contract. Historical rule: the face limiter is the guarantee iff interior marked cells = 0 on the deciding run_hydro run, 4000 cycles as in #263, on both CPU and CUDA. Marks drive redos, so #263's 4000 cycles / 0 redos already settles the CPU decision; CUDA is the new deciding evidence in P. The CPU per-cell and mass-budget table still needs the study instrumentation; zero redos does not count round-off clamps or ghost-cell added mass.
```

## grid, decision, open items

```text
Superseded by rev 9: retain only the IM and MM z=1 columns of the historical grid below; z!=1 and the old A/B and dropped unit-test requirements are inactive.
Grid, target test_flux_positivity_carry. Columns: IM cpu/cuda | MM z=1 cpu/cuda | MM z!=1 cpu/cuda. E = exists at 5eeb9b6 (line), N = new, G = gap today (gap test line).
- adv lmars: E169/NG | G351/NG | N/N
- adv hllc: E169/NG | NG/NG | N/N
- settling: E184/NG | NG/NG | N/N
- along x2: E197/NG | NG/NG | N/N
- donors: E238/E242 | NG/NG | N/N
- mixed: E334/E339 | NG/NG | N/N
- A cell is covered only if it ran on its device and passed; CPU-only or skip = gap.
- Each moist-mixture cell runs for both A and B.
- The z!=1 cells use the test kintera build. The settling cells isolate the cloud (h_c = u_c, which is unchanged by z).

Tests added
- The parameterised Carry.<case>/<eos> and Carry_cuda.<case>/<eos>.
- Unit tests: I1, I2, I3, X, A-NI, G-AB, and the kintera function tests (1b).

Superseded by rev 9: the entire historical A/B decision below is replaced by the 12:58 DECIDE and its linked rev 9b per-test accuracy criteria; the W->E bug remains its own fix.
Decision (branch a)
- Both A and B remain supported formulations. Neither is removed and no TORCH_CHECK is added.
- A formulation is eligible as the default if it passes every gate in its scope.
- B must pass O1, O2, I1-I4 and all 24 moist-mixture cells.
- A must pass I1, I2, G-AB, A-NI, the carry mechanics and all z=1 cells. On the z!=1 cells, A's miss against O1 is reported, not a rejection (7.6e-6 dry, 1.5e-4 vapour at the card).
- Choose the default by, in order:
  1. The z!=1 O1/O2 agreement, if the default must cover real gases.
  2. CUDA parity.
  3. Cost (B <= A + 15%).
  4. LOC.
- Condensates: K1 (u_c) unless the EOS gains a condensate volume.
- The W->E bug is always its own fix, never folded into the winner.
- The kintera test option and eval_intEng_R_ddC go in a kintera PR with its own numbers. The test functions stay test-only.

Open
Superseded by rev 9: the following kintera PR ownership task is dropped; non-ideal work goes to a Limit line and a separate upstream issue for human discussion.
- Xi to confirm who owns the kintera PR: the test-only option plus eval_intEng_R_ddC.
- Not verified:
  - harp's element table: kintera computes M via harp::get_compound_weight (molar_mass.cpp:17-18); IUPAC weights give 28.96998e-3 or 28.96968e-3;
  - that the Newton T solve in thermo_y.cpp converges for z!=1 at the card state (the python estimate gives T 353.440 K).
- The baseline counts are only a prior until Gate 0.
```
