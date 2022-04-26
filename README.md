# Query2particles



The code is for the implementation of the paper *Query2Particles: Knowledge Graph Reasoning with Particle Embeddings*, 
accepted by the Findings of NAACL 2022. 


Before running the code please download the knowledge graph data from [here](http://snap.stanford.edu/betae/KG_data.zip)
, and put them into <code> ./data </code> folder. For the details of the data, please check [here](https://github.com/snap-stanford/KGReasoning).

This code includes running two benchmark datasets for evaluating FOL and EPFL queries respectively. 
To run the code, first you need to change the permissions of the shell scripts:

<code>chmod u+x *.sh </code>

To train and evaluate the results on FOL queries, you can run the following scripts separately:
<code>./run_beta_benchmark_FB15K.sh </code>
<code>./run_beta_benchmark_FB15K_237.sh </code>
<code>./run_beta_benchmark_NELL.sh </code>


To train and evaluate the results on EPFL queries, you can run the following scripts separately:
<code>./run_q2p_benchmark_FB15K.sh 
</code>
<code>./run_q2p_benchmark_FB15K_237.sh </code>
<code>./run_q2p_benchmark_NELL.sh </code>

You can also use tensorboard to monitor the training and evaluating process. To start the tensorboard,
please use
<code> tensorboard --logdir ./q2p_logs/gradient_tape/the-directory-named-by-starting-time --port the-port-you-fancy </code>.








