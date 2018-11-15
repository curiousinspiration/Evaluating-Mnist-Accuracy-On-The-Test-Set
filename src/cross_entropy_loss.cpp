/*
 * CrossEntropyLoss Implementation
 */

#include "neural/loss/cross_entropy_loss.h"

#include <glog/logging.h>

#include <sstream>
#include <math.h>

using namespace std;

namespace neural
{

CrossEntropyLoss::CrossEntropyLoss()
{

}

float CrossEntropyLoss::Forward(
    const TTensorPtr& a_inputs, const TTensorPtr& a_targets) const
{
    if (a_inputs->Shape().size() != 2 || a_targets->Shape().size() != 2)
    {
        stringstream l_ss;
        l_ss << "CrossEntropyLoss::Forward shape size 2 "
             << a_inputs->ShapeStr() << " != " << a_targets->ShapeStr() << endl;
        throw(runtime_error(l_ss.str()));
    }

    /*
    Example:
    # batch size 2, predictions
    inputs(2x4) = [
        [0.1, 0.2, 0.65, 0.05],
        [0.75, 0.05, 0.1, 0.1]
    ]
    # batch size 2, one hot encoding for the classees
    targets(2,1) = [
        0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    ]
    */

    /*
    -1/n * sum((y * log(yhat)) + ((1-y) * log(1-yhat)))
    */

    size_t l_batchSize = a_inputs->Shape().at(0);
    size_t l_inputSize = a_inputs->Shape().at(1);

    float l_error = 0.0;
    for (size_t i = 0; i < l_batchSize; ++i)
    {
        for (size_t j = 0; j < l_inputSize; ++j)
        {
            float y = a_targets->At({i,j});
            float yhat = a_inputs->At({i,j});
            l_error += (y * log(yhat)) + ((1.0 - y) * log(1.0 - yhat));
            // LOG(INFO) << "y: " << y << " yhat: " 
            //           << yhat << " error: " << l_error << endl;
        }
    }

    return -1.0 * (l_error / (float)l_batchSize);
}

TTensorPtr CrossEntropyLoss::Backward(
    const TTensorPtr& a_origInputs, const TTensorPtr& a_targets)
{
    TMutableTensorPtr l_gradient = Tensor::New(a_targets->Shape());

    size_t l_batchSize = a_origInputs->Shape().at(0);
    size_t l_inputSize = a_origInputs->Shape().at(1);
    for (size_t i = 0; i < l_batchSize; ++i)
    {
        for (size_t j = 0; j < l_inputSize; ++j)
        {
            float yhat = a_origInputs->At({i,j});
            float y = a_targets->At({i,j});
            
            // derivative of ln(x) is 1/x
            // https://www.wyzant.com/resources/lessons/math/calculus/derivative_proofs/lnx
            // float l_gradVal = -1.0 * ((y / yhat) + ((1.0 - yhat) * (1.0 / (1.0 - yhat))));
            float l_gradVal = yhat;
            if (y == 1)
            {
                l_gradVal -= 1;
            }
            l_gradient->SetAt({i,j}, l_gradVal);
        }
    }

    return make_shared<Tensor>(*l_gradient /= (float) l_batchSize);
}

} // namespace neural
