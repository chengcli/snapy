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
#setup_cuda_test(test_thomas_solver)
