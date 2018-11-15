/*
 * SoftmaxLayer Implementation
 */

#include "neural/layers/softmax_layer.h"
#include "neural/math/tensor_math.h"

#include <glog/logging.h>

#include <iostream>
#include <sstream>
#include <utility>
#include <thread>

#include <math.h>

using namespace std;

namespace neural
{

SoftmaxLayer::SoftmaxLayer()
{

}

TTensorPtr SoftmaxLayer::Forward(const TTensorPtr& a_inputs) const
{
    if (0 == a_inputs->Shape().size())
    {
        return a_inputs;
    }

    if (a_inputs->Shape().size() > 2)
    {
        stringstream l_ss;
        l_ss << "SoftmaxLayer::Forward size > 2 not supported yet size = " 
             << a_inputs->Shape().size() << endl;
        throw(runtime_error(l_ss.str()));
    }

    // Input will be batchsize x pred size
    // sum up the exp(s) so we can divide each one by the sum
    // x00 = math.exp(1.0) / (math.exp(1.0) + math.exp(2.0) + math.exp(3.0))

    size_t x = a_inputs->Shape().at(0);
    size_t y = a_inputs->Shape().at(1);

    // For numeric stability we subtract max from input
    // because exp(x) can get very large, but by subtracting the max
    // we guaruntee max == 0
    // see http://cs231n.github.io/linear-classify/#softmax
    TMutableTensorPtr l_inputs = a_inputs->ToMutable();
    *l_inputs -= a_inputs->MaxVal();

    TMutableTensorPtr l_outputs = Tensor::New({x, y});

    vector<float> l_sums;
    for (size_t i = 0; i < x; ++i)
    {
        float l_sum = 0.0;
        for (size_t j = 0; j < y; ++j)
        {
            l_sum += exp(l_inputs->At({i, j}));
        }
        // cout << "sum: " << l_sum << endl;
        l_sums.push_back(l_sum);
    }

    for (size_t i = 0; i < x; ++i)
    {
        for (size_t j = 0; j < y; ++j)
        {
            float l_val = exp(l_inputs->At({i, j})) / l_sums.at(i);
            // cout << "Setting exp(l_inputs->At({" << i << ", " << j << "})) " 
            //      << l_inputs->At({i, j}) << " / l_sums.at(" << i << "): " 
            //      << exp(l_inputs->At({i, j})) << "/" << l_sums.at(i)
            //      << " = " << l_val << endl;
            l_outputs->SetAt({i, j}, l_val);
        }
    }

    return l_outputs;
}

TTensorPtr SoftmaxLayer::Backward(
    const TTensorPtr& a_origInput, const TTensorPtr& a_gradOutput)
{
    return a_gradOutput;
}

} // namespace neural
