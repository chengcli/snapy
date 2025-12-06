#pragma once

#include <torch/torch.h>

/*!
 * Global cartesian coordinates
 * ----------------------------
 *
 *       +Z
 *       ^
 *       |
 *       |----> +Y
 *      /
 *  +X /
 *
 */

namespace snap {

//! Transform cubed sphere velocity from panel 1 to panel 2
//! \param a $x = \tan(\xi)$ coordinates
//! \param b $y = \tan(\eta)$ coordinat
torch::Tensor vel_zab_from_p1(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel);

torch::Tensor vel_zab_from_p2(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel);

torch::Tensor vel_zab_from_p3(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel);

torch::Tensor vel_zab_from_p4(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel);

torch::Tensor vel_zab_from_p5(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel);

torch::Tensor vel_zab_from_p6(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel);

}  // namespace snap
