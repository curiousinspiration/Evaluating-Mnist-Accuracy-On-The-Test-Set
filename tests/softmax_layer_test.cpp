/*
 * Softmax Layer Test
 *
 */

#include "neural/layers/softmax_layer.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(SoftmaxTest, TestForward)
{
    SoftmaxLayer l_layer;
    TTensorPtr l_input = Tensor::New(
        {2,3},
        {
            1.0, 2.0, 3.0,
            3.0, 4.0, 7.0
        }
    );

    TTensorPtr l_output = l_layer.Forward(l_input);

    vector<size_t> l_shape = l_output->Shape();
    EXPECT_EQ(2, l_shape.at(0));
    EXPECT_EQ(3, l_shape.at(1));

    /*
    # in python
    import math
    x00 = math.exp(1.0) / (math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
    
    [[0.0900, 0.2447, 0.6652],
     [0.0171, 0.0466, 0.9362]]
    */
    EXPECT_NEAR(0.0900, l_output->At({0, 0}), 0.0001f);
    EXPECT_NEAR(0.2447f, l_output->At({0, 1}), 0.0001f);
    EXPECT_NEAR(0.6652f, l_output->At({0, 2}), 0.0001f);
    EXPECT_NEAR(0.0171f, l_output->At({1, 0}), 0.0001f);
    EXPECT_NEAR(0.0466f, l_output->At({1, 1}), 0.0001f);
    EXPECT_NEAR(0.9362f, l_output->At({1, 2}), 0.0001f);
}

TEST(SoftmaxTest, TestNumericStability)
{
    SoftmaxLayer l_layer;

    // math.exp(789) will overflow, so we want to make sure our 
    // Softmax can handle this
    TTensorPtr l_input = Tensor::New(
        {1,3},
        {
            759.0, 760.0, 761.0
        }
    );

    TTensorPtr l_output = l_layer.Forward(l_input);

    vector<size_t> l_shape = l_output->Shape();
    EXPECT_EQ(1, l_shape.at(0));
    EXPECT_EQ(3, l_shape.at(1));

    EXPECT_NEAR(0.09003057f, l_output->At({0, 0}), 0.0001f);
    EXPECT_NEAR(0.24472847f, l_output->At({0, 1}), 0.0001f);
    EXPECT_NEAR(0.66524096f, l_output->At({0, 2}), 0.0001f);
}

