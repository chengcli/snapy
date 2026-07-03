import snapy


def test_hydro_options_exposes_fused_recon_riemann():
    options = snapy.HydroOptions()

    assert options.fused_recon_riemann() is False
    assert options.fused_recon_riemann(True) is options
    assert options.fused_recon_riemann() is True
    assert options.fused_recon_riemann(False) is options
    assert options.fused_recon_riemann() is False
