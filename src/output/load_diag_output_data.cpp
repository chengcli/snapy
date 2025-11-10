// fmt
#include <fmt/format.h>

// kintera
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/snap.h>

#include "output_type.hpp"
#include "output_utils.hpp"

namespace snap {

void OutputType::loadDiagOutputData(MeshBlockImpl* pmb, Variables const& vars) {
  if (!(pmb->phydro->options.eos().type() == "ideal-moist" ||
        pmb->phydro->options.eos().type() == "moist-mixture")) {
    // skip if not using kintera thermo
    return;
  }

  OutputData* pod;
  auto peos = pmb->phydro->peos;

  auto const& w = vars.at("hydro_w");

  auto m = pmb->named_modules()["hydro.eos.thermo"];
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);
  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(w.device());

  int ny = thermo_y->options.species().size() - 1;
  auto temp = peos->compute("W->T", {w});
  auto pres = w[IPR];
  auto xfrac = thermo_y->compute("Y->X", {w.narrow(0, ICY, ny)});

  // mole concentration [mol/m^3]
  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

  // volumetric entropy [J/(m^3 K)]
  auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});

  // volumetric heat capacity [J/(m^3 K)]
  auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

  // molar entropy [J/(mol K)]
  auto entropy_mole = entropy_vol / conc.sum(-1);

  // molar heat capacity [J/(mol K)]
  auto cp_mole = cp_vol / conc.sum(-1);

  // mean molecular weight [kg/mol]
  auto mu = (thermo_x->mu * xfrac).sum(-1);

  // specific entropy [J/(kg K)]
  auto entropy = entropy_mole / mu;

  // potential temperature [K]
  auto theta = (entropy_vol / cp_vol).exp();

  // relative humidity
  auto rh = kintera::relative_humidity(temp, conc, thermo_x->stoich,
                                       thermo_x->options.nucleation());

  if (ContainVariable("diag")) {
    // temperature
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "temp";
    pod->data.CopyFromTensor(temp);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // potential temperature
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "theta";
    pod->data.CopyFromTensor(theta);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // entropy
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "entropy";
    pod->data.CopyFromTensor(entropy);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // relative humidity
    auto reactions = thermo_x->options.nucleation().reactions();
    for (int i = 0; i < reactions.size(); ++i) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = fmt::format("rh_{}", reactions[i].products().begin()->first);
      pod->data.CopyFromTensor(rh.select(-1, i));

      AppendOutputDataNode(pod);
      num_vars_ += 1;
    }

    // implicit corrrection
    if (pmb->phydro->options.imp().scheme() > 0) {
      auto du = pmb->phydro->named_buffers()["M"];

      // density
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "ic_dry";
      pod->data.InitFromTensor(du, 4, Index::IDN, 1);
      AppendOutputDataNode(pod);
      num_vars_++;

      // momentum
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "ic_mom";
      pod->data.InitFromTensor(du, 4, Index::IVX, 3);

      AppendOutputDataNode(pod);
      num_vars_ += 3;

      // total energy
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "ic_etot";
      pod->data.InitFromTensor(du, 4, Index::IPR, 1);

      AppendOutputDataNode(pod);
      num_vars_++;

      // vapor + cloud
      auto ny = peos->nvar() - 5;
      if (ny > 0) {
        pod = new OutputData;
        pod->type = "VECTORS";
        pod->name = get_hydro_names(pmb, "ic_");
        pod->data.InitFromTensor(du, 4, Index::ICY, ny);

        AppendOutputDataNode(pod);
        num_vars_ += ny;
      }
    }
  }
}
}  // namespace snap
