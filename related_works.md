
* Weak LTH, Pruning+Training 
	* [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf): Learn only the "important" connections. Combination of pruning \& training. This can be considered as finding tickets
	* [Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf): Neural networks contain sparse subnetworks that can be effectively trained from scratch when reset to their initialization

	* [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/pdf/1905.01067.pdf): NOTE: Sec.5 has some technique using probabilistic masking. We need to compare it with ours


* Strong LTH, (only) Pruning
	* [What’s Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/pdf/1911.13299.pdf): Randomly weighted (overparameterized) neural network contains a subnetwork which performs near SOTA. Suggested Edge-popup (EP) algorithm.

* Sparsity usinig L0 regularization

	* [LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION](https://arxiv.org/pdf/1712.01312.pdf): Suggested "surrogate" L0 regularization, in order to sparsify NN. [Q] Not sure how they applied "reparameterization trick"
	
	* [Winning the Lottery with Continuous Sparsification](https://arxiv.org/pdf/1912.04427.pdf)

	* [SNIP: SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY](https://arxiv.org/pdf/1810.02340.pdf)



* Optimization in real/binary values
	* [Pseudo-boolean optimization](https://www.sciencedirect.com/science/article/pii/S0166218X01003419)


* Mode connectivity
	* [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/pdf/1912.05671.pdf)
	* [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/pdf/1802.10026.pdf)
	* [Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes](https://arxiv.org/abs/2104.11044)
		- MLI property: linear interpolation from initial to final neural net params typically decreases the loss monotonically
		- Proved that MLI property holds with high probability for networks of sufficient width
		- Proved that small "Gauss length" gives monotonicity
		- Kind of related with "laze training" saying that training goes to the closest minima?
		- NOTE: the connectivity is quite good. See what's happening


* To Be Categorized
	* [Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://arxiv.org/pdf/2009.11094.pdf)
	* [Pruning Neural Networks at Initialization: Why are We Missing the Mark?](https://arxiv.org/pdf/2009.08576.pdf)
	* [Supermasks in Superposition](https://proceedings.neurips.cc//paper/2020/file/ad1f8bb9b51f023cdc80cf94bb615aa9-Paper.pdf): Superposition of supermasks can be used for target-task inference \& continual learning?

	* [PICKING WINNING TICKETS BEFORE TRAINING BY PRESERVING GRADIENT FLOW](https://openreview.net/pdf?id=SkgsACVKPH)


