/*
 * Example tool training feed forward neural network on mnist data
 *
 */


#include "neural/data/mnist_dataloader.h"
#include "neural/layers/linear_layer.h"
#include "neural/layers/relu_layer.h"
#include "neural/loss/squared_error_loss.h"

#include <math.h>

#include <glog/logging.h>

using namespace neural;
using namespace std;

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

float CalcAccuracy(
    LinearLayer& a_firstLayer,
    ReLULayer& a_secondLayer,
    LinearLayer& a_thirdLayer,
    MNISTDataloader& a_testDataloader)
{
    float l_numCorrect = 0.0;
    float l_numTotal = 0.0;
    size_t l_numIters = a_testDataloader.DataLength();
    for (size_t i = 0; i < l_numIters; ++i)
    {
        TMutableTensorPtr l_input, l_output;
        a_testDataloader.DataAt(i, l_input, l_output);
        float l_targetOutput = l_output->At({0,0});

        // Forward pass
        TTensorPtr l_output0 = a_firstLayer.Forward(l_input);
        TTensorPtr l_output1 = a_secondLayer.Forward(l_output0);
        TTensorPtr l_predTensor = a_thirdLayer.Forward(l_output1);
        float l_predVal = l_predTensor->At({0,0});

        if (round(l_predVal) == l_targetOutput)
        {
            l_numCorrect += 1.0;
        }

        if (i % 1000 == 0)
        {
            LOG(INFO) << "Processing test set... " << i << endl;
        }
        l_numTotal += 1.0;
    }

    float l_accuracy = (l_numCorrect / l_numTotal) * 100;
    LOG(INFO) << "Accuracy = " << l_numCorrect << "/" << l_numTotal 
              << " = " << l_accuracy << "%" << endl;
    return l_accuracy;
}

int main(int argc, char const *argv[])
{
    // Define data loader
    string l_dataPath = "../data/mnist/";
    MNISTDataloader l_trainDataloader(l_dataPath, true); // second param for isTrain?
    MNISTDataloader l_testDataloader(l_dataPath, false); // second param for isTrain?

    // Define model
    // first linear layer is 784x300
    // 784 inputs, 300 hidden size
    LinearLayer firstLinearLayer(Tensor::Random({784, 300}, -0.01f, 0.01f));

    // Non-linear activation
    ReLULayer activationLayer;
    
    // second linear layer is 300x1
    // 300 hidden units, 1 output
    LinearLayer secondLinearLayer(Tensor::Random({300, 1}, -0.01f, 0.01f));

    // Error function
    SquaredErrorLoss loss;

    // Training loop
    float learningRate = 0.0001;
    size_t numEpochs = 10;
    size_t numIters = l_trainDataloader.DataLength();
    for (size_t i = 0; i < numEpochs; ++i)
    {
        LOG(INFO) << "--EPOCH (" << i << ")--" << endl;
        vector<float> errorAcc;
        for (size_t j = 0; j < numIters; ++j)
        {
            // Get training example
            TMutableTensorPtr input, output;
            l_trainDataloader.DataAt(j, input, output);
            float targetOutput = output->At({0,0});

            // Forward pass
            TTensorPtr output0 = firstLinearLayer.Forward(input);
            TTensorPtr output1 = activationLayer.Forward(output0);
            TTensorPtr y_pred = secondLinearLayer.Forward(output1);
            float yPredVal = y_pred->At({0,0});

            // Calc Error
            float error = loss.Forward(yPredVal, targetOutput);
            errorAcc.push_back(error);

            // Only log every 1000 examples
            if (j % 1000 == 0)
            {
                float avgError = CalcAverage(errorAcc);
                LOG(INFO) << "--ITER (" << i << "," << j << ")-- avgError = " << avgError << " lr = " << learningRate << endl;
                LOG(INFO) << "Got prediction: " << yPredVal << " for target " << targetOutput << endl;
                errorAcc.clear();
            }

            // Backward pass
            float errorGrad = loss.Backward(yPredVal, targetOutput);
            TTensorPtr y_predGrad = secondLinearLayer.Backward(output1, Tensor::New({1,1}, {errorGrad}));
            TTensorPtr grad1 = activationLayer.Backward(output0, y_predGrad);
            TTensorPtr grad0 = firstLinearLayer.Backward(input, grad1);

            // Gradient Descent
            secondLinearLayer.UpdateWeights(learningRate);
            firstLinearLayer.UpdateWeights(learningRate);

            if (j % 10000 == 0)
            {
                CalcAccuracy(firstLinearLayer, activationLayer, secondLinearLayer, l_testDataloader);
            }
        }
        CalcAccuracy(firstLinearLayer, activationLayer, secondLinearLayer, l_testDataloader);
        learningRate /= 2.0;
    }

    return 0;
}
