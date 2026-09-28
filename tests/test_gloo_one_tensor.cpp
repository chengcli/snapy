// C/C++
#include <string>
#include <vector>

// gtest
#include <gtest/gtest.h>

// snap
#include <snap/layout/layout.hpp>
#include <snap/layout/process_group.hpp>

using namespace snap;

// Two ranks on Gloo, which sends one tensor per message (#240): a message of
// one tensor goes through, and a message of two is refused before it reaches
// Gloo, with a message that names the fix.
TEST(gloo, a_message_of_two_tensors_is_refused_with_the_fix) {
  auto opts = LayoutOptionsImpl::create();  // rank and port from torchrun
  opts->backend("gloo");
  if (opts->process_world_size() != 2) GTEST_SKIP() << "needs 2 processes";
  auto comm = ProcessGroupContext::create(opts);
  int rank = opts->process_rank(), peer = 1 - rank;

  std::vector<torch::Tensor> one = {torch::full({3}, 1. + rank)};
  std::vector<torch::Tensor> got = {torch::zeros({3})};
  auto sent = comm->send(one, peer, 7);
  comm->recv(got, peer, 7)->wait();
  sent->wait();
  EXPECT_TRUE(torch::equal(got[0], torch::full({3}, 1. + peer)));

  std::vector<torch::Tensor> two = {torch::zeros({3}), torch::zeros({3})};
  for (bool is_send : {true, false}) {
    try {
      is_send ? comm->send(two, peer, 8) : comm->recv(two, peer, 8);
      ADD_FAILURE() << (is_send ? "send" : "recv") << " of two tensors passed";
    } catch (std::exception const& e) {
      std::string msg = e.what();
      EXPECT_NE(msg.find("one tensor per message"), std::string::npos) << msg;
    }
  }
}
