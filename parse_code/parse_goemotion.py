import pandas as pd
import numpy as np
import json
from parse_gab_hate_speech import calculate_entropy


with open('goemotion_data/sentiment_mapping.json', 'r') as f:
    SENTIMENT_MAP = json.load(f)


def find_most_disagreed_samples():
    df1 = pd.read_csv('goemotion_data/goemotions_1.csv')
    df2 = pd.read_csv('goemotion_data/goemotions_2.csv')
    df3 = pd.read_csv('goemotion_data/goemotions_3.csv')
    annotations = pd.concat([df1, df2, df3], axis=0)

    def emotion_to_sentiment(row):
        positive = False
        neutral = False
        negative = False
        for emotion in SENTIMENT_MAP['positive']:
            if row[emotion] == 1:
                positive = True
        for emotion in SENTIMENT_MAP['negative']:
            if row[emotion] == 1:
                negative = True
        if row['neutral'] == 1:
            neutral = True
        else:
            for emotion in SENTIMENT_MAP['ambiguous']:
                if row[emotion] == 1:
                    neutral = True

        return pd.Series([positive, negative, neutral])

    annotations[['positive', 'negative', 'ambiguous']] = annotations.apply(emotion_to_sentiment, axis=1)
    print(annotations.info())

    # Count the number of annotations
    annotation_num = annotations['id'].value_counts()
    print(annotation_num.value_counts())

    # # If there are 3 annotation, disagreement might be caused by annotation error instead of ambiguity
    # # So we keep those with more than 3 annotations
    # valid_annotation_id = annotation_num[annotation_num > 3].index.tolist()
    # annotations_filtered = annotations.loc[annotations['id'].isin(valid_annotation_id), :]
    #
    # # Get highly disagreed annotations
    # annotation_entropy = annotations_filtered.groupby('id')['Hate'].apply(calculate_entropy)
    # high_entropy_idx = annotation_entropy[annotation_entropy > 0.9].index.tolist()
    # annotations_high_entropy = annotations_filtered.loc[annotations_filtered['ID'].isin(high_entropy_idx), :]
    # annotations_high_entropy.to_csv('gab_hate_speech_data/parsed/high_entropy_annotations.csv', index=False)
    #
    # # Randomly sample some agreed annotations
    # all_annotation_entropy = annotations.groupby('ID')['Hate'].apply(calculate_entropy)
    # low_entropy_idx = all_annotation_entropy[all_annotation_entropy < 0.1].index.tolist()
    # annotations_low_entropy = annotations.loc[annotations['ID'].isin(low_entropy_idx), :].sample(n=len(annotations_high_entropy), random_state=42)
    # annotations_low_entropy.to_csv('gab_hate_speech_data/parsed/low_entropy_annotations.csv', index=False)


if __name__ == '__main__':
    find_most_disagreed_samples()