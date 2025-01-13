import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


# Define a function to calculate entropy for a group
def calculate_entropy(series):
    probabilities = series.value_counts(normalize=True)  # Get probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities))  # Compute entropy
    return entropy


def find_heterogeneous_annotator():
    annotator_beliefs = pd.read_csv('gab_hate_speech_data/AnnotatorIAT_and_Attitudes.csv')
    cosine_sim = cosine_similarity(annotator_beliefs.drop(columns=['PostsAnnotated']).values)
    # Find the flattened index of the minimum value
    flat_index = np.argmin(cosine_sim)
    # Convert the flattened index to 2D coordinates
    row, col = np.unravel_index(flat_index, cosine_sim.shape)
    print(annotator_beliefs['Annotator'][row], annotator_beliefs['Annotator'][col])
    difference_scores = 1 - cosine_sim.mean(axis=1)
    # Add difference scores to the dataframe
    annotator_beliefs['difference_score'] = difference_scores
    # Sort rows by difference score (most different first)
    sorted_df = annotator_beliefs.sort_values(by='difference_score', ascending=False)
    # Show results
    print(sorted_df)


def find_most_disagreed_samples():
    annotations = pd.read_csv('gab_hate_speech_data/GabHateCorpus_annotations.tsv', sep='\t')
    # Count the number of annotations
    annotation_num = annotations['ID'].value_counts()
    print(annotation_num.value_counts())

    # If there are 3 annotation, disagreement might be caused by annotation error instead of ambiguity
    # So we keep those with more than 3 annotations
    valid_annotation_id = annotation_num[annotation_num > 3].index.tolist()
    annotations_filtered = annotations.loc[annotations['ID'].isin(valid_annotation_id), :]

    # Get highly disagreed annotations
    annotation_entropy = annotations_filtered.groupby('ID')['Hate'].apply(calculate_entropy)
    high_entropy_idx = annotation_entropy[annotation_entropy > 0.9].index.tolist()
    annotations_high_entropy = annotations_filtered.loc[annotations_filtered['ID'].isin(high_entropy_idx), :]
    annotations_high_entropy.to_csv('gab_hate_speech_data/parsed/high_entropy_annotations.csv', index=False)

    # Randomly sample some agreed annotations
    all_annotation_entropy = annotations.groupby('ID')['Hate'].apply(calculate_entropy)
    low_entropy_idx = all_annotation_entropy[all_annotation_entropy < 0.1].index.tolist()
    annotations_low_entropy = annotations.loc[annotations['ID'].isin(low_entropy_idx), :].sample(n=len(annotations_high_entropy), random_state=42)
    annotations_low_entropy.to_csv('gab_hate_speech_data/parsed/low_entropy_annotations.csv', index=False)


# Count disagreement between all pairs of annotators. Save disagreement between (annotator_1, annotator_2)
def disagree_pairs(annotator_1, annotator_2):
    df = pd.read_csv('gab_hate_speech_data/GabHateCorpus_annotations.tsv', sep='\t')
    # Create an empty dictionary to store disagreement counts
    disagreement_ids = {}

    # Group by 'ID'
    grouped = df.groupby("ID")

    # Iterate over each group
    for idx, group in grouped:
        # Generate all pairs of annotators
        annotator_pairs = list(combinations(group["Annotator"], 2))

        # Iterate over each pair
        for annotator1, annotator2 in annotator_pairs:
            # Get labels for each annotator
            label1 = group.loc[group["Annotator"] == annotator1, "Hate"].values[0]
            label2 = group.loc[group["Annotator"] == annotator2, "Hate"].values[0]

            # Check if there's a disagreement
            if label1 != label2:
                # Use a sorted tuple as the key to ensure consistency
                pair = tuple(sorted((annotator1, annotator2)))
                if pair in disagreement_ids.keys():
                    disagreement_ids[pair].append(idx)
                else:
                    disagreement_ids[pair] = []

    disagreement_counts = {k: len(v) for k, v in disagreement_ids.items()}
    # Convert the disagreement counts to a DataFrame for better readability
    disagreement_df = pd.DataFrame(
        list(disagreement_counts.items()),
        columns=["Annotator Pair", "Disagreements"]
    )
    disagreement_df = disagreement_df.sort_values(by=['Disagreements'], ascending=False)
    disagreement_df.to_csv('gab_hate_speech_data/parsed/disagree_count.csv')

    target_pair = tuple(sorted((annotator_1, annotator_2)))
    data_points = []
    label_1_list = []
    label_2_list = []
    for sample_id in disagreement_ids[target_pair]:
        data_points.append(df.loc[df['ID'] == sample_id, "Text"].iloc[0])
        label_1_list.append(df.loc[(df['ID'] == sample_id) & (df['Annotator'] == target_pair[0]), "Hate"].iloc[0])
        label_2_list.append(df.loc[(df['ID'] == sample_id) & (df['Annotator'] == target_pair[1]), "Hate"].iloc[0])
    disagrees = pd.DataFrame({"Text": data_points, f"Annotator_{target_pair[0]}": label_1_list, f"Annotator_{target_pair[1]}": label_2_list})
    disagrees.to_csv(f'gab_hate_speech_data/parsed/disagrees_between_{target_pair[0]}_{target_pair[1]}.csv')


if __name__ == '__main__':
    disagree_pairs(0, 11)
    find_heterogeneous_annotator()
