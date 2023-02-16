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

If you think this code is useful, please cite the original paper. 

```
@inproceedings{bai-etal-2022-query2particles,
    title = "{Q}uery2{P}articles: Knowledge Graph Reasoning with Particle Embeddings",
    author = "Bai, Jiaxin  and
      Wang, Zihao  and
      Zhang, Hongming  and
      Song, Yangqiu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.207",
    doi = "10.18653/v1/2022.findings-naacl.207",
    pages = "2703--2714",
    abstract = "Answering complex logical queries on incomplete knowledge graphs (KGs) with missing edges is a fundamental and important task for knowledge graph reasoning. The query embedding method is proposed to answer these queries by jointly encoding queries and entities to the same embedding space. Then the answer entities are selected according to the similarities between the entity embeddings and the query embedding. As the answers to a complex query are obtained from a combination of logical operations over sub-queries, the embeddings of the answer entities may not always follow a uni-modal distribution in the embedding space. Thus, it is challenging to simultaneously retrieve a set of diverse answers from the embedding space using a single and concentrated query representation such as a vector or a hyper-rectangle. To better cope with queries with diversified answers, we propose Query2Particles (Q2P), a complex KG query answering method. Q2P encodes each query into multiple vectors, named particle embeddings. By doing so, the candidate answers can be retrieved from different areas over the embedding space using the maximal similarities between the entity embeddings and any of the particle embeddings. Meanwhile, the corresponding neural logic operations are defined to support its reasoning over arbitrary first-order logic queries. The experiments show that Query2Particles achieves state-of-the-art performance on the complex query answering tasks on FB15k, FB15K-237, and NELL knowledge graphs.",
}
```




