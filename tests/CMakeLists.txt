# ==========
# Unit Tests
# ==========

enable_testing()

#setup_test(test_eos)
setup_test(test_weno)
setup_test(test_reconstruct)
setup_test(test_riemann)
setup_test(test_internal_boundary)
setup_test(test_rectify)
setup_test(test_pull_2d)
setup_test(test_pull_3d)
setup_test(test_slab)
setup_test(test_cubed)
setup_test(test_cubed_sphere)
setup_test(test_exchange_buffers)
#setup_test(test_read_topo)
#setup_cuda_test(test_thomas_solver)
#setup_test(test_aneos)
