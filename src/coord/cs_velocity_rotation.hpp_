#pragma once

#include <torch/torch.h>

//! Panel number assiangments
//! (1) Top
//! (2) Front
//! (3) Left
//!
//!                                           z
//!                ___________                |
//!                |\        .\           x___| (1)
//!                | \   1   . \              /
//!                |  \_________\           y/
//!                | 3 |     .  |
//!                \. .|......  |
//! z___ (3)        \  |    2 . |
//!    /|            \ |       .|             y
//!   / |             \|________|             |
//!  y  x                                 (2) |___x
//!                                          /
//!                                        z/

//! Panel number assiangments
//! (4) Right
//! (5) Bottom
//! (6) Back
//!                                        y  x
//!                __________              | /
//!                |\       .\             |/___z
//!                | \      . \           (4)
//!     y  z       |  \________\
//!     | /        |  |  6  .  |
//! x___|/         |..|......  |
//!      (6)       \  |     . 4|       (5) ___ x
//!                 \ |  5   . |          /|
//!                  \|_______.|         / |
//!                                     y  z

namespace snap {

//! Transform cubed sphere velocity to local cartesian velocity
torch::Tensor vel_zab_to_zxy(torch::Tensor vel, torch::Tensor a,
                             torch::Tensor b);

//! Transform local cartesian velocity to cubed sphere velocity
torch::Tensor vel_zxy_to_zab(torch::Tensor vel, torch::Tensor a,
                             torch::Tensor b);

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
