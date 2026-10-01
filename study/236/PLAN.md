# #236 study plan, rev 7 (branch (a))

Study plan for issue #236; not for merge. Drafting only; nothing run.

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
- f. P had no decision rule -> the face limiter is the guarantee iff marked cells = 0 on the run_hydro run.
- g. L79 and the grid cited lines of 842a116, which no chengcli ref holds -> they cite 5eeb9b6 lines. The mapping found two 842a116 tests that do not exist at 5eeb9b6: moist_mixture_withheld_mass_keeps_its_energy_and_momentum (the grid's MM z=1 lmars cell is now the gap test at 5eeb9b6:351) and moist_mixture_nasa9_h2_enthalpy_matches_internal_plus_pressure (the hand h_dry test; a new test does this).

Rev 5 critique -> change
- a. Gate 0 built main 3f7ad96, which is not current main -> Gate 0 builds current main 5eeb9b6 (v2.10.36, kintera >= 2.5.13); the 3f7ad96 counts are a prior only, and a count mismatch is recorded, not an abort.
- b. Gate 4 used 842a116 as a bitwise oracle -> Gate 4 = full ctest plus two bitwise checks, both against 5eeb9b6: (1) ideal-moist fluxes, (2) formulation A with no extras; 842a116 is a candidate only and never the oracle; no rebase comparison.
- c. The non-ideal option's species scope was implicit -> the test-only option registers virial z and u for dry gas and vapour ONLY and leaves every cloud extra unset; this scope is stated in the option text, in O1 and in Gate 1b.
- d. Where species positivity is guaranteed was not studied -> new item P: face flux limiter vs cell clamp, what the cell clamp does to mass, energy and momentum, and what it reports; reproducer is the 2D moist Jupiter CRM with limiter: true that clamps vapour/cloud on nearly every step (#263).

Issue: https://github.com/chengcli/snapy/issues/236

## critique -> change, scope

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

```text
kintera main (4dc613d) today
- func2 is f(T, c_n) per species; call_func2 adds to a preset value (utils_dispatch.cpp:47-67, CPU; utils_dispatch.cu:42-102, CUDA).
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
- I1: sum over dry, vapour and cloud of c_n M_n h_n = U + p + rho KE. Tolerance 1e-6 J/m3 + 1e-12. At O1 the residual is 3.7e-9 on |U+p| = 2.17e7.
- I2: h_n in cell i is unchanged, bitwise, when the neighbours change. h - KE is unchanged when v changes, within 4 eps S_n.
- I3: (dU/dV)_T,N = T (dp/dT)_V,N - p, by Richardson finite difference on the kintera VT->U and VT->P path. Tolerance 1e-3 Pa + 1e-8. At O1: 690600.0 = 690600.0, residual 2.2e-5 Pa.
- I4: B = O2 within 1e-6 J/mol + 1e-8.

Dry gas left out in rev 3, now fixed
- The sum identity: a new test computes h_dry with the A formula by hand (842a116's nasa9 test did this; 5eeb9b6 has no such test). Under B it must use B's dry value.
- G-AB covered only the vapour and the cloud; the dry row is added.
- The column totals covered only IPR and IVX-IVZ; IDN and every ICY row are added, exact to the same tolerance.
- The counts: the dry mass flux on vs off must be exactly equal, on every face, CPU and CUDA.
- O3 had no h_d; it is added (part 3).
- The dropped inv_mu control for dry was listed only for the sum; it is now also a per-species gate.
```

## oracles and controls

```text
B (J/mol, then divide by M_n)
- zeta = z + c z'; vbar_n = zeta_n / sum c zeta; phi_n = c_n (u_n' + R T z_n').
- Gas: h_n = u_n + z_n R T + phi_n - vbar_n sum c phi.
- Cloud: h_c = u_c.
- A = u_n + z_n R T.
- Assumptions: p = R T sum c z(T, c_n); clouds have zero volume; fixed T and p; T = 0 reference with u0_R from the YAML. R = 8.314462618.

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
- T = p / (R sum_gas rho y / M)
- h_d = (R/M_d)(cv_R T + T) + KE
- h_v = (R/M_v)(u0_R + cv_R T + T) + KE
- h_c = (R/M_c)(u0_R + cv_R T) + KE
- Card: dry 2.5; vapour 3.5, u0 0; cloud 9.0, u0 -3430. M comes from the code at Gate 0 (nominal 28.97e-3 / 18.015e-3).
- Values: T 353.347 K; h_d 3.54940e5, h_v 7.33861e5, h_c -1.15325e5 J/kg.

Controls, each must miss by >= 1e3 x its tolerance
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
- T vs code: 1e-9 K / 1e-11
- h_n vs O1 or O3, all species including dry: 1e-9 J/kg / 1e-12
- O2 vs O1: 1e-6 J/mol / 1e-8
- I1: 1e-6 J/m3 / 1e-12
- I3: 1e-3 Pa / 1e-8
- limited counts vs Gate 0, and the dry mass flux: exact
- face energy residual: 1e-9 / 1e-12
- face momentum residual: 1e-12 / 1e-12
- column totals IDN, ICY.., IPR, IVX-IVZ: 1e-12 / 1e-12 of sum|du|
- CPU vs CUDA:
  - h_n: 1e-9 / 1e-14
  - theta: 1e-15 / 1e-13
  - species flux: 1e-15 / 1e-13
  - energy flux: 1e-9 / 1e-13
  - momentum flux: 1e-12 / 1e-13
  - counts: exact
- code M vs composition M: rel 2e-5
- runtime: median of 5; soft gate B <= A + 15%

G-AB (z=1 cells only)
- |h_A - h_B| <= 4 eps S_n, eps = 2^-52, S_n = (R/M)(|u0_R| + cv_R T + z T) + KE.
- It applies to dry, vapour and cloud, including ghosts.
- With no extras registered, czh = 1, czh_ddC = 0 and extra = 0 exactly (eval_uhs.cpp:225-226, 247, 268-288), so B = A + 0.
- Bound at the card: cloud 2.7e-9, vapour 6.5e-10 J/kg.
- It covers the 12 z=1 cells.
- If it fails, B is not the claimed reduction and is stopped.

Gates
- 0 (CPU): build current main 5eeb9b6 (v2.10.36) against kintera v2.5.15 (4dc613d, kintera main), the same for both builds so the bitwise checks reproduce. Record the six moist-mixture counts, M_n and T. The earlier 3f7ad96 counts are a prior only. Pass if all six moist-mixture cases are red (energy residual rel 1.0 at every limited face) and every ideal-moist arm is green. A count mismatch against the prior is recorded, not an abort. 842a116's after-numbers are recorded as a candidate's, not as an oracle.
- 1 (CPU, no snapy): O1/O2 are reproduced in J/mol and J/kg, including the card z!=1 values. I1 and I3 hold on O1. Every control misses by its stated size (+-10%) and by >= 1e3 x tolerance, X included.
- 1b (CPU and CUDA, kintera test build): scope is dry gas and vapour ONLY; every cloud extra must read back unset (czh = 1, czh_ddC = 0, extra = 0 exactly). The registered functions match O1 per species: czh, czh_ddC, eval_intEng_R and eval_intEng_R_ddC to 1e-12. I3 holds on the VT->U and VT->P path. X, registered in the same build, fails I3.
- 2 (CPU): 18 cells x {A, B} meet every tolerance. z=1 cells go against O3; z!=1 cells against O1 at each cell's state and O2. I1 and I2 hold for A and B. A-NI holds. G-AB holds on 6 cells. The controls fail.
- 3 (CUDA, fp64): 18 cells x {A, B} meet the same checks plus the CPU vs CUDA rows. G-AB holds on 6 cells. A skip is a gap and fails the gate.
- 4: full ctest (+ python limiter tests) shows no new failures vs 5eeb9b6 except moist_mixture_withholds_no_energy_or_momentum_yet, replaced by expect_carried(off, on) as its comment directs, plus two bitwise checks, both against 5eeb9b6: (1) ideal-moist fluxes are bitwise equal; (2) formulation A with no extras is bitwise equal to 5eeb9b6 in every flux row on every unlimited face (a face where every species' donor theta == 1, i.e. share == 0; flux_positivity.cpp:127-136 on 5eeb9b6), and in the species rows on limited faces, for one forward pass. 842a116 is a candidate only, never the oracle. No rebase comparison. Column totals hold. Runtime recorded.
- Coverage: all 36 cells ran on their device and passed.
```

## positivity: face flux limiter vs cell clamp (new in rev 5)

```text
P. Where is species positivity guaranteed?
- Two places today: the face flux limiter (limits the species flux so no donor goes negative) and the cell repair in apply_conserved_limiter_ (equation_of_state.cpp:253-354 on 5eeb9b6, in step order: parent borrow, a parented cloud borrows its deficit from its parent vapour in the same cell (:269-285); parentless fix_vapor + clamp_min_(0) on the clouds (:302-328); vapour pass via columnar fix_vapor with TORCH_CHECK (:330-344); marks (:348-353)).
- Question: which one is the guarantee? If the face limiter guarantees it, the cell clamp should fire only at round-off; if it does not, the clamp is the real guarantee and its side effects are part of the scheme.

What the cell clamp does (to be measured, per clamped cell and per column)
- Mass: the parent borrow and fix_vapor conserve mass; mass is added only by clamp_min_(0) on a parentless cloud with a negative column total (delta = -min(rho_c,0)). Record the added mass and the mass moved between cells.
- Energy: the clamp changes species without changing IEN, so the implied T and p shift. Record the change in total energy vs the energy the added mass would carry (h_c or h_v at the cell state).
- Momentum: IVX-IVZ are untouched, so the velocity rho v / rho changes when rho changes. Record the momentum and KE change.
- Reporting: the clamp sets limiter_marks_[0] only when a change exceeds positivity_roundoff * rho (#256); below that it is silent. Record how many cells are clamped, how many are marked, and the largest |delta|/rho, per step.

Reproducer
- 2D moist Jupiter CRM (H2O + NH3, 100x100, ideal-moist, limiter: true, rk3, cfl 0.9) from #263; with the external driver (path+sha recorded) species repairs above round-off occur nearly every step; with run_hydro, 0 redos.
- Context from #263 (closed, not a snapy source defect): with the external driver applying kinetics to a stale hydro_w, ~30k cells start each step with negative species (min/rho -8.8e-13) and every redo is a species clamp; with hydro_w refreshed first, 4000 cycles run with 0 redos and the negatives are denormal (min/rho -1.1e-303); #263's run_hydro row (#257 order) is 4000 cycles, 0 redos, and "-" for negative species at step start, so the denormal negatives were measured only with the refreshed external driver. Measured on the 5eeb9b6 tree (fa136b5), CPU.
- Whether a species-only clamp should request a redo (#226's choice) is decided here, not in #263.

Pass for P: a table per run of clamped cells, marked cells, max |delta|/rho, and the mass, energy and momentum change per step, CPU and CUDA; and a statement of which mechanism is the positivity guarantee, by this rule: the face limiter is the guarantee iff marked cells = 0 on the run_hydro run, on both CPU and CUDA.
```

## grid, decision, open items

```text
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
- Xi to confirm who owns the kintera PR: the test-only option plus eval_intEng_R_ddC.
- Not verified:
  - harp's element table: kintera computes M via harp::get_compound_weight (molar_mass.cpp:17-18); IUPAC weights give 28.96998e-3 or 28.96968e-3;
  - that the Newton T solve in thermo_y.cpp converges for z!=1 at the card state (the python estimate gives T 353.440 K).
- The baseline counts are only a prior until Gate 0.
```
