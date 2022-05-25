This is the repository for the paper <a href='https://arxiv.org/abs/2205.04652'> SuMe: A Dataset Towards Summarizing Biomedical Mechanisms </a>

Here you can use files to pretrain, train, and evaluate SuMe dataset. For each model (BART, GPT2, T5) we have different preprocessing steps. The training is similar for all of them.
Use `train.py` in each folder to train the model, with `generation.py` you can generate the output and the `postprocessing.py` will compute different metrics introduced in the paper to evaluate the generation. 
