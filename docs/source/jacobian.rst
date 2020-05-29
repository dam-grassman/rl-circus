Jacobian Saliency Map
=====================

Paper : *Dueling Network Architectures for Deep Reinforcement Learning*, Whang et al 2016

In the paper that introduces the Dueling Network Architectures for DQN, the authors added a short section on saliency maps visualization using a gradient-based method (see *Simonyan et al., 2013*).	
They do so by computing the **absolute value of the Jacobian of the neural network w.r.t the input frames : :math:`\nabla_s \hat{V}(s,\theta)`
