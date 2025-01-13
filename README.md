# Annotation Variance
In this project, we explore the research question:
- When annotating classification data, can LLMs identify datapoints that might be systematically disagreed among human annotators? I.e., the ambiguous samples with high human annotation variance.

## Datasets
We gather datasets with multiple human annotations.
#### Dataset 1: Gab Hate Speech Corpus

Features:
- 27k datapoints, each with 3+ annotations from 18 annotators.
- Demographic / attitude features of 8 annotators.
- Labels of binary (hate speech or not) and finer granularity.
- Paper: https://link.springer.com/article/10.1007/s10579-021-09569-x

Raw Files:
- `gab_hate_speech_data/AnnotatorIAT_and_Attitudes.csv`: Features of annotators.
- `gab_hate_speech_data/GabHateCorpus_annotations.tsv`: All annotations from various annotators.

Parsed Files:
- `gab_hate_speech_data/parsed/disagree_count.csv`: Number of disagreement between each pair of annotators.
- `gab_hate_speech_data/parsed/disagrees_between_0_11.csv`: Disagreed samples of Anno_1 and Anno_11.
- `gab_hate_speech_data/parsed/high_entropy_annotations.csv`: Controversial / hard datapoints where annotators highly disagree with each other.
- `gab_hate_speech_data/parsed/low_entropy_annotations.csv`: Randomly sample datapoints with less disagreement.

Parsing Code:
- `parse_code/parse_gab_hate_speech.py`: Find the differences between annotators (e.g., Anno_0 and Anno_11). Find annotations with high entropy. Find disagreements between annotators.

#### Dataset 2: GoEmotions

Features:
- 27 granular categories of emotions, belonging to 6 general emotions and 3 sentiments.
- Has a field `example_very_unclear` where annotators mark difficulty.
- 3+ annotations for 58k datapoints.
- Paper: https://arxiv.org/abs/2005.00547

Parsing Code:
- `parse_code/parse_goemotion.py`




