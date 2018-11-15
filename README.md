# Evaluating Mnist Accuracy On The Test Set

How to quantitatively know how well your neural network is generalizing.

Code to go along with blog: [Evaluating Mnist Accuracy On The Test Set](http://www.curiousinspiration.com/posts/evaluating-mnist-accuracy-on-the-test-set)

# Build

`mkdir build`

`cd build`

`cmake -D BLAS_INCLUDE_DIR=/usr/local/opt/openblas/include \
       -D BLAS_LIB_DIR=/usr/local/opt/openblas/lib \
       -D GLOG_INCLUDE_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/include/ \
       -D GLOG_LIB_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/lib/ \
       -D GTEST_INCLUDE_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/include/ \
       -D GTEST_LIB_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/lib/ ..`

`make`

`./tests`

`./feedforward_neural_net`