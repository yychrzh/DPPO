# dppo
# my simple implementation single thearding ppo alogrithm based on tensorflow
the implementation is test on gym games
Dependencies:
tensorflow r1.3
gym 0.9.2

Proximal policy optimization Algorithms, has some of the benefits of trust region policy optimization(TRPO), but it is much simpler to implement for TRPO is a constained problem and the alogrithm need to use conjugate gradient algorithm to updata the policy's parameter, 
which need to construct the Fisher information matrix.


