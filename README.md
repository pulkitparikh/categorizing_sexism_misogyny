We hereby release the code used for our research paper under review at TWEB titled "Categorizing Sexism and Misogyny through Neural Approaches". Our implementation utilizes parts of the code from [1, 2, 3] and libraries Keras and Scikit-learn [4]. The following are brief descriptions of some of the contents of this repository. Any further information about the code can be obtained by emailing me (my email id is mentioned in the paper).

1) main.py

- The main file that needs to be run for all deep learning based methods including the proposed approach and baselines

2) neural_approaches.py

- Training, prediction, evaluation, training data creation/transformation, loss function assignment, class imbalance correction

3) dl_models.py

- Deep learning architectures for the proposed approach as well as baselines

4) load_pre_proc.py

- Data loading, pre-processing, problem transformation, functions wrt our ensemble method, and other utilities

5) sent_enc_embed.py

- Generation of sentence representations using general-purpose sentence encoding schemes

6) word_embed.py

- Generation of distributional word representations

7) ling_word_feats.py 

- Generation of a linguistic/aspect-based word-level representation

8) gen_batch_keras.py 

- Generation of batches of inputs for training and testing

9) auto_encode.py

- Functions related to the autoencoder-based method for using unlabeled data and the pre-training of BERT on a domain-specific corpus (esp. around data creation) 

10) eval_measures.py 

- Functions related to multi-label evaluation and result reporting

11) traditional_ML.py 

- Traditional machine learning methods on ngram based and other features

12) doc2vec_embed.py 

- Creation of a vector representation of a piece of text using doc2vec 

13) rand_approach.py 

- Random label assignment in accordance with normalized training frequencies of labels

14) rand_sample.py

- Creation of a small random sample of the data for quick experimentation

15) split_labels.py

- Label subset generation for our ensemble approach

16) att_visualize.py

- Functions used for quantitative and qualitative analysis

17) config_deep_learning.txt

- A sample configuration file for deep learning methods specifying multiple nested and non-nested parameter combinations

18) config_traditional_ML.txt

- A sample configuration file for traditional machine learning methods

References:

[1] Sweta Agrawal and Amit Awekar. 2018. Deep learning for detecting cyberbullying across multiple social media platforms. In European Conference on Information Retrieval. Springer, 141–153.

[2] Richard Liao. 2017. textClassifier. https://github.com/richliao/textClassifier.

[3] Nikhil Pattisapu, Manish Gupta, Ponnurangam Kumaraguru, and Vasudeva Varma. 2017. Medical persona classification in social media. In Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017. ACM, 377–384.

[4] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12 (2011), 2825–2830.
