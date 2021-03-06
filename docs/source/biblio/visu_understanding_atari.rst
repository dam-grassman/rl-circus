Visualizing and Understanding Atari Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tag** : XRL, 

**Article** : *Visualizing and Understanding Atari Agents*, Sam Greydanus, Anurag Koul, Jonathan Dodge and Alan Fern, **September 2018**.

**code**: https://github.com/greydanus/visualize_atari (MIT License)

.. image:: 
    /biblio/media/breakout_tunneling.gif
    :width: 200
.. image:: 
    /biblio/media/pong_killshot.gif
    :width: 200
.. image:: 
    /biblio/media/spaceinv_aiming.gif
    :width: 200
    

**Contribution** : method for generating useful *saliency maps* to show :

    (1) what the agent attends to
    (2) whether the agent makes its decision for the right or wrong reasons  
    (3) how the agent evolves during learning.

describe a simple perturbation-based technique for generating saliency videos of deep RL agents (poor quality of Jacobian saliency) which aims at visualizing and understanding the policies of any DRL that uses visual input.

Goal : show that saliency information provide signigicant and reliable insight to understand the RL agent's decisions.

Poor quality of Jacobian saliency maps compare to the new proposed method : 

.. image:: 
    /biblio/media/comparison_jacobian.png
    :width: 500
