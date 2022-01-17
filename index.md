## What is SuMe?
### SuMe is a dataset towards Summarizing Biomedical Mechanisms.

<!-- You can find our paper [here](Link)  -->

Mohaddeseh Bastan, Mihai Surdeanu, Niranjan Balasubramanian. 

We introduce the SuMe dataset, the first dataset towards summarizing biomedical mechanisms and the underlying relations between entities. The dataset contains 22K mechanism summarization instances collected semi-automatically, an evaluation partition of 125 instances that were corrected by domain experts. We also create a conclusion generation task from the larger set of 611K abstracts which we use as a pretraining task for mechanism generation models.

### Example
In the following example we see an example of an entry in the SuMe dataset. Some supporting text was removed to save space. The input is the supporting sentences with the main two entities. The output is the relation type and a sentence concluding the mechanism underlying the relationship.

<!-- <img src="https://github.com/MHDBST/SuMe/blob/main/Dataexample.drawio-18.pdf" alt="Image of PerSenT stats"/> -->
<!-- <a href="https://github.com/MHDBST/SuMe/blob/main/Dataexample.drawio-18.pdf"></a> -->
<!-- <script src="https://github.com/MHDBST/SuMe/blob/main/Dataexample.drawio-18.pdf"></script> -->
<!-- <a class="js-navigation-open Link--primary" title="Dataexample.drawio-18.pdf" data-pjax="#repo-content-pjax-container" href="https://github.com/MHDBST/SuMe/blob/main/Dataexample.drawio-18.pdf">Dataexample.drawio-18.pdf</a> -->
<!-- <div role="rowheader" class="flex-auto min-width-0 col-md-2 mr-3">
                                                        <span class="css-truncate css-truncate-target d-block width-fit">
                                                            <a class="js-navigation-open Link--primary" title="Dataexample.drawio-18.pdf" data-pjax="#repo-content-pjax-container" href="https://github.com/MHDBST/SuMe/blob/main/Dataexample.drawio-18.pdf">Dataexample.drawio-18.pdf</a>
                                                        </span>
                                                    </div> -->
                                                    
<object data="/assets/pdf/example.pdf" width="1000" height="1000" type='application/pdf'></object>

### Dataset Statistics
We construct SuMe using biomedical abstracts from the PubMed open access subset. Starting from 1.1M scientific papers, we followed the following sequence of bootstrapping steps to prepare the SuMe dataset. 
1. Finding Conclusion Sentences
2. Extracting Main Entities & Relation. We run biomedical relation extractor, REACH which can identify entities and the relations between entities.
3. Filtering for Mechanism Sentences
We separate out the abstracts for which the conclusion sentences are predicted to have non-mechanism related conclusions as additional related data that can be use for pretraining the generation models we eventually train for the mechanism summarization task. Dataset Statistics: Each dataset contains a number of unique abstracts, a supporting set, a mechanism sentence a pair of entities. The first entity is called the regulator entity (regulator) and the second one is called the regulated entity (regulated)
<object data="/assets/pdf/stats.pdf" width="1000" height="1000" type='application/pdf'></object>


### Download the data
<!-- You can download the data set URLs from [here](https://github.com/MHDBST/PerSenT/blob/main/train_dev_test_URLs.pkl)

The processed version of the dataset which contains used paragraphs, document-level, and paragraph-level labels can be download separately as [train](https://github.com/MHDBST/PerSenT/blob/main/train.csv), [dev](https://github.com/MHDBST/PerSenT/blob/main/dev.csv), [random test](https://github.com/MHDBST/PerSenT/blob/main/random_test.csv), and [fixed test](https://github.com/MHDBST/PerSenT/blob/main/fixed_test.csv).

To recreat the results from the paper you can follow the instructions in the readme file from the [source code](https://github.com/StonyBrookNLP/PerSenT/tree/main/pre_post_processing_steps). -->

### Liked us? Cite us!

<!--  Please use the following bibtex entry:

   ```
@inproceedings{bastan2020authors,
      title={Author's Sentiment Prediction}, 
      author={Mohaddeseh Bastan and Mahnaz Koupaee and Youngseo Son and Richard Sicoli and Niranjan Balasubramanian},
      year={2020},
      eprint={2011.06128},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
   ``` -->
