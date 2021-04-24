
* Weak LTH, Pruning+Training 
	* Learning both Weights and Connections for Efficient Neural Networks (2015)
		- Learn only the "important" connections. Combination of pruning \& training.
	* Lottery Ticket Hypothesis
		- Neural networks contain sparse subnetworks that can be effectively trained from scratch when reset to their initialization

	* Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask
		- NOTE: Sec.5 has some technique using probabilistic masking. We need to compare it with ours


* Strong LTH, (only) Pruning
	* What’s Hidden in a Randomly Weighted Neural Network? 
		- Randomly weighted (overparameterized) neural network contains a subnetwork which performs near SOTA. 
		- Edge-popup (EP) algorithm

* Sparsity usinig L0 regularization

	* LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION
		- Suggested "surrogate" L0 regularization, in order to sparsify NN
		- Q. Not sure how they applied "reparameterization trick"
	
	* Winning the Lottery with Continuous Sparsification

	* [SNIP: SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY](https://arxiv.org/pdf/1810.02340.pdf)

	* 

* Optimization in real/binary values
	* Pseudo-boolean function


* Mode connectivity
	* Linear Mode Connectivity and the Lottery Ticket Hypothesis
	* Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
	* [Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes](https://arxiv.org/abs/2104.11044)
		- Linear interpolation from initial to final neural net params typically decreases the loss monotonically
		- NOTE: the connectivity is quite good. See what's happening


* To Be Categorized
	* Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot
	* [Pruning Neural Networks at Initialization: Why are We Missing the Mark?](https://arxiv.org/pdf/2009.08576.pdf)
	* Supermasks in Superposition
		- Superposition of supermasks can be used for target-task inference \& continual learning?

	* [PICKING WINNING TICKETS BEFORE TRAINING BY PRESERVING GRADIENT FLOW](https://openreview.net/pdf?id=SkgsACVKPH)


