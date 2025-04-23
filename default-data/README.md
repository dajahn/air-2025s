# Subtask 4b (Scientific Claim Source Retrieval)

Given an implicit reference to a scientific paper, i.e., a social media post (tweet) that mentions a research publication without a URL, retrieve the mentioned paper from a pool of candidate papers.

The task is a retrieval task, where the query set contains tweets mentioning papers implicitly, and the collection set contains the pool of papers mentioned by tweets.

__Table of contents:__

<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Task 1 Directory](#contents-of-the-repository)
- [Datasets statistics](#datasets-statistics)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)

<!-- ## Evaluation Results

TBA -->

## List of Versions
- [15/01/2025] Training data released.

## Contents of the Subtask4b Directory

This repository contains the following files:

1. **getting_started_subtask4b.ipynb** Jupyter notebook that enables participants of subtask 4b to quickly get started
2. **subtask4b_collection_data.pkl** The collection set for the retrieval task (CORD-19 academic papers' metadata)
3. **subtask4b_query_tweets_train.tsv** The query set (train split) for the retrieval task (tweets with implicit references to CORD-19 papers)
4. **subtask4b_query_tweets_dev.tsv** The query set (dev split) for the retrieval task (tweets with implicit references to CORD-19 papers)
4. **README.md** this file

## Datasets statistics
The data for the retrieval task is made of two sets:
* **Query set (train & dev splits)**
  * content: tweets with implicit references to scientific papers from CORD-19.
  * stats: 14,253 tweets.
* **Collection set**
  * content: metadata of the CORD-19 scientific papers which tweets implicitly refer to.
  * stats: metadata of 7,764 scientific papers.

## Input Data Format
### Query set (train & dev splits)

The tweets are provided as a TSV file with three columns:
> post_id <TAB> tweet_text <TAB> cord_uid

Where: <br>
* post_id: a unique identifier for the tweet. Ids were generated for easier evaluation of submissions
* tweet_text: the tweets' texts
* cord_uid: the label to be predicted, which is the unique identifier of the scientific paper that the tweet refers to

**Example:**
See Subtask4b examples at: https://checkthat.gitlab.io/clef2025/task4/

### Collection set
The metadata of CORD-19 papers are provided as a pickle file with the following columns:

> cord_uid <TAB> title <TAB> abstract <TAB> authors <TAB> journal <TAB> [...] (17 metadata columns in total)

**Example:**
See Subtask4b examples at: https://checkthat.gitlab.io/clef2025/task4/

## Output Data Format

The output must be a TSV format with two columns: 
> post_id <TAB> preds

where the "preds" column contains an array of the top5 predicted cord_uids (in descending order: [pred1, pred2, pred3, pred4, pred5], where pred1 is the cord_uid of the highest ranked publication). Please use the provided notebook which contains code to generate the submission file in the correct format. The notebook is available here: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task4/subtask_4b (see "getting_started_subtask4b.ipynb")

## Evaluation Metrics

This task is evaluated as a retrieval task. We will use the Mean Reciprocal Rank (MRR@5) to evaluate it. 

## Baselines & Scorers

A "Getting started" jupyter notebook can be found in this repository. Participants can download it and use it to get started quickly. The notebook contains:

1. Code to upload data, including:
   * code to upload the collection set (CORD-19 academic papers' metadata)
   * code to upload the query set (tweets with implicit references to CORD-19 papers)
2. Code to run a baseline retrieval model (BM25)
3. Code to evaluate the baseline model using the MRR score


## Submission

Please find the submission details on the Codalab page of the task: https://codalab.lisn.upsaclay.fr/competitions/22359

## Related Work

Information regarding the annotation guidelines can be found in the following document: https://github.com/AI-4-Sci/ReferenceDisambiguation/blob/main/annotation_protocol_examples.pdf 


## Contact
Please contact salim.hafid@sciencespo.fr.

## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task4/
