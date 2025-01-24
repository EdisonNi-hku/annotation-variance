import numpy as np
import pandas as pd
import requests
import json
import math
import uuid
import openai
import matplotlib.pyplot as plt

from typing import List, Annotated
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from collections import Counter
from itertools import combinations

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

openai.api_key = 'OPENAI_API_KEY'

ghc_path = 'GabHateCorpus_annotations.tsv'
dis_path = 'GabHateCorpus/disagrees_between_0_11.csv'

ghc = pd.read_csv(ghc_path, sep = '\t')
dis = pd.read_csv(dis_path)

"""**Types of hate speech**
- “Assaults on Human Dignity” (**HD**)
- “Calls for Violence” (**CV**)
- “Vulgarity and/or Offensive Language” (**VO**)

## Modeling Annotation Entropy
"""

ghc_ent = ghc.groupby('ID').agg({
    'Text': 'first',  # Keep the first occurrence of the text as they are the same for each ID
    'Hate': list,      # Aggregate the 'Hate' column into a list
    'HD': list,
    'CV': list,
    'VO': list,
    'ID': 'first',
    'Annotator': list,
})

def calculate_entropy(row):
    annotations = row['Hate']
    total = len(annotations)

    if total == 0:
        return 0  # No annotations, so entropy is 0

    count_0 = annotations.count(0)
    count_1 = annotations.count(1)

    # Proportions
    p0 = count_0 / total
    p1 = count_1 / total

    # Calculate entropy
    entropy = 0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)

    return entropy

# Apply the function to each row
ghc_ent['Hate_entropy'] = ghc_ent.apply(calculate_entropy, axis=1)

ghc_ent = ghc_ent[ghc_ent['Hate'].apply(lambda x: len(x) > 4)]
ghc_ent['Hate_pos'] = ghc_ent['Hate'].apply(lambda x: np.mean(x))
ghc_ent['Hate_neg'] = ghc_ent['Hate'].apply(lambda x: 1 - np.mean(x))

"""### S1 Ask for Distribution"""

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a team of helpful research assistants with expertise in detecting hate speech from social media posts.

        Objective: You are tasked with evaluating social media posts to assess the distribution of labels assigned by annotators regarding whether a post constitutes hate speech. The goal is to predict the percentage of annotators who classify the post as hate speech and those who classify it as not hate speech, based on the specific aspects of hate speech it may exhibit.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each post, provide a brief reasoning for your prediction, focusing on the clarity or ambiguity of the post's content concerning aspects of hate speech.
        - Then, output a list of two floating-point numbers indicating the percentages, e.g., [0.2, 0.8]."""

        p_user = f"""Here is the post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotation_percentage(BaseModel):

    """Detect Annotator Disagreements on Hate Speech Posts"""
    Percentages: str = Field(description=f"A list of two floating-point numbers indicating the percentages.")
    Reasoning: str = Field(description=f"Reasoning for the prediction, focusing on the clarity or ambiguity of the post's content concerning the aspects of hate speech.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotation_percentage)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotation_percentage"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'ID': df['ID'].iloc[i],
            'Text': df['Text'].iloc[i],
            'Hate_pos': df['Hate_pos'].iloc[i],
            'Hate_neg': df['Hate_neg'].iloc[i],
            'Hate_pos_LLM': eval(output['Percentages'])[0],
            'Hate_neg_LLM': eval(output['Percentages'])[1],
            'Reasoning_LLM': output['Reasoning'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(ghc_ent)
output = get_output(prompts, ghc_ent, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'ID': post_value['ID'],
        'Text': post_value['Text'],
        'Hate_pos': post_value['Hate_pos'],
        'Hate_neg': post_value['Hate_neg'],
        'Hate_pos_LLM': post_value['Hate_pos_LLM'],
        'Hate_neg_LLM': post_value['Hate_neg_LLM'],
        'Reasoning_LLM': post_value['Reasoning_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'ID',
    'Text',
    'Hate_pos',
    'Hate_neg',
    'Hate_pos_LLM',
    'Hate_neg_LLM',
    'Reasoning_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

# Sanity check
df[(df['Hate_pos_LLM'] + df['Hate_neg_LLM']) != 1].shape

df['Hate_pos_diff'] = abs(df['Hate_pos'] - df['Hate_pos_LLM'])

"""### S2 Ask for Diversity"""

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a team of helpful research assistants with expertise in detecting hate speech from social media posts.

        Objective: You are tasked with evaluating social media posts to assess the labels assigned by annotators regarding whether a post constitutes hate speech. The goal is to first generate ten plausible opinions that a human annotator might hold when assessing the following post, based on specific aspects of hate speech it may exhibit. Then, based on each opinion, assign a label: 1 for hate speech and 0 for non-hate speech. Finally, return two lists: (1) ten distinct opinions and (2) the ten corresponding labels.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each post, first provide ten plausible opinions that a human annotator might hold when assessing the post, based on specific aspects of hate speech it may exhibit, and return a list thereof.
        - Then, based on each opinion, assign a label: 1 for hate speech and 0 for non-hate speech; return a list of the ten corresponding labels."""

        p_user = f"""Here is the post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_opinions_and_annotations(BaseModel):

    """Detect opinions and annotations on Hate Speech Posts"""
    Opinions: list = Field(description=f"A list of ten plausible opinions that a human annotator might hold when assessing the post.")
    Labels: list = Field(description=f"A list of ten labels based on the opinions; 1 for hate speech and 0 for non-hate speech.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_opinions_and_annotations)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_opinions_and_annotations"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'ID': df['ID'].iloc[i],
            'Text': df['Text'].iloc[i],
            'Hate_pos': df['Hate_pos'].iloc[i],
            'Opinions_LLM': output['Opinions'],
            'Labels_LLM': output['Labels'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(ghc_ent)
output = get_output(prompts, ghc_ent, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'ID': post_value['ID'],
        'Text': post_value['Text'],
        'Hate_pos': post_value['Hate_pos'],
        'Opinions_LLM': post_value['Opinions_LLM'],
        'Labels_LLM': post_value['Labels_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'ID',
    'Text',
    'Hate_pos',
    'Opinions_LLM',
    'Labels_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

# Sanity check
df[df['Opinions_LLM'].apply(lambda x: len(x) != 10)].shape

df['Hate_pos_LLM'] = df['Labels_LLM'].apply(lambda x: np.mean(x))
df['Hate_diff'] = abs(df['Hate_pos'] - df['Hate_pos_LLM'])

"""### S2.1 Knowledge vs. Simulation"""

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a helpful research assistant with expertise in detecting hate speech from social media posts.

        Objective: You are tasked with evaluating social media posts to assess the distribution of labels assigned by annotators regarding whether a post constitutes hate speech. The goal is to first generate a list of plausible opinions that a human annotator might hold when assessing the following post, based on specific aspects of hate speech it may exhibit. Then, based on the opinions, predict the percentage of annotators who classify the post as hate speech and those who classify it as not hate speech.
        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each post, first provide a list plausible opinions that a human annotator might hold when assessing the post, based on specific aspects of hate speech it may exhibit.
        - Then, based on the opinions, output a list of two floating-point numbers indicating the percentages, e.g., [0.2, 0.8]."""

        p_user = f"""Here is the post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotation_percentage(BaseModel):

    """Detect Annotator Disagreements on Hate Speech Posts"""
    Opinions: list = Field(description=f"A list of the plausible opinions a human annotator might hold when assessing the post.")
    Percentages: str = Field(description=f"A list of two floating-point numbers indicating the percentages.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotation_percentage)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotation_percentage"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'ID': df['ID'].iloc[i],
            'Text': df['Text'].iloc[i],
            'Hate_pos': df['Hate_pos'].iloc[i],
            'Hate_pos_LLM': eval(output['Percentages'])[0],
            'Hate_neg_LLM': eval(output['Percentages'])[1],
            'Opinions_LLM': output['Opinions'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(ghc_ent)
output = get_output(prompts, ghc_ent, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'ID': post_value['ID'],
        'Text': post_value['Text'],
        'Hate_pos': post_value['Hate_pos'],
        'Hate_pos_LLM': post_value['Hate_pos_LLM'],
        'Hate_neg_LLM': post_value['Hate_neg_LLM'],
        'Opinions_LLM': post_value['Opinions_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'ID',
    'Text',
    'Hate_pos',
    'Hate_pos_LLM',
    'Hate_neg_LLM',
    'Opinions_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

df['Hate_diff'] = abs(df['Hate_pos'] - df['Hate_pos_LLM'])

df['Opinions_LLM'].apply(lambda x: len(x)).value_counts()

df['Opinions_LLM'].apply(lambda x: len(x)).mean()

"""### S3 Annotator Modeling"""

ghc_dis = ghc[ghc['Annotator'].isin([0, 11])]

n = 55
k = 5
# 1) n positive samples annotated by annotator 0
pos_a0 = ghc_dis[(ghc_dis['Annotator'] == 0) & (ghc_dis['Hate'] == 1)].sample(n=n, random_state=42)
pos_a0 = pos_a0.sample(frac=1, random_state=42).reset_index(drop=True)

# 2) n negative samples annotated by annotator 0
neg_a0 = ghc_dis[(ghc_dis['Annotator'] == 0) & (ghc_dis['Hate'] == 0)].sample(n=n, random_state=42)
neg_a0 = neg_a0.sample(frac=1, random_state=42).reset_index(drop=True)

# 3) n positive samples annotated by annotator 11
pos_a11 = ghc_dis[(ghc_dis['Annotator'] == 11) & (ghc_dis['Hate'] == 1)].sample(n=n, random_state=42)
pos_a11 = pos_a11.sample(frac=1, random_state=42).reset_index(drop=True)

# 4) n negative samples annotated by annotator 11
neg_a11 = ghc_dis[(ghc_dis['Annotator'] == 11) & (ghc_dis['Hate'] == 0)].sample(n=n, random_state=42)
neg_a11 = neg_a11.sample(frac=1, random_state=42).reset_index(drop=True)

ghc_dis_a0_dev = pd.concat([pos_a0.head(k), neg_a0.head(k)]).sample(frac=1, random_state=42).reset_index(drop=True)
ghc_dis_a11_dev = pd.concat([pos_a11.head(k), neg_a11.head(k)]).sample(frac=1, random_state=42).reset_index(drop=True)

ghc_dis_a0_test = pd.concat([pos_a0.tail(n-k), neg_a0.tail(n-k)]).sample(frac=1, random_state=42).reset_index(drop=True)
ghc_dis_a11_test = pd.concat([pos_a11.tail(n-k), neg_a11.tail(n-k)]).sample(frac=1, random_state=42).reset_index(drop=True)

text = ghc_dis_a0_dev['Text'].tolist()
lbl = ghc_dis_a0_dev['Hate'].tolist()

# Initialize an empty string to store the merged paragraph
paragraph_a0, paragraph_a11 = "", ""

# Iterate through each row in the dataframe
for index, row in ghc_dis_a0_dev.iterrows():
    post = row['Text']
    label = row['Hate']
    paragraph_a0 += f" Post: {post} | Label: {label} \n"

print(paragraph_a0)

for index, row in ghc_dis_a11_dev.iterrows():
    post = row['Text']
    label = row['Hate']
    paragraph_a11 += f" Post: {post} | Label: {label} \n"

print(paragraph_a11)

"""#### A0"""

### Annotator_0

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a helpful research assistant with expertise in detecting hate speech from social media posts.

        The task is to evaluate social media posts and predict the labels assigned by human annotator Andy, determining whether a post constitutes hate speech based on specific aspects of hate speech it may exhibit. Samples labeled by Andy are provided for reference. Carefully review these samples and consider how Andy would label the target post, and provide the label for it.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each target post, first provide a brief reasoning on how Andy would label it, based on specific aspects of hate speech it may exhibit.
        - Then, output an integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.

        Sample posts annotated by Andy:
        Post: When was the last time we saw a leader who loves his country and its people? | Label: 0
        Post: Everything bad in America for the last 83 years is because of Jews. Smash these fucking rat-faced kikes and purge them from America. | Label: 1
        Post: No it's not ok with me,what I meant,we knew almost everything what was in the memo,now what? We need to see somebody going go jail if that was you or me we could be in jail by now. | Label: 0
        Post: Thug-ette and the Flab Four appearing tonight only at Applebee's. | Label: 1
        Post: Great time to be alive! | Label: 0
        Post: "You can thank #BHO & his wolves for #USA's division & lawlessness"  | Label: 1
        Post: How does MURDER make you free??? | Label: 0
        Post: Given the opportunity, I would be tempted to corner, subdue and ass-rape Brian Kilmeade. | Label: 1
        Post: Newsweek made a point to assure the bitch bun wearing beta cucks and feminazi's that GAB is an isolated social media outlet with a small audience. Fast forward to a year from now when GAB will have users in the hundreds of thousands.They'll be crying by the masses and screaming at the sky.   | Label: 1
        Post: Chick on the left looks like that old Bengals WR T.J Houshmandzadeh | Label: 0 """

        p_user = f"""Here is the target post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotator_label(BaseModel):

    """Detect Annotator Andy's label on Hate Speech Posts"""
    Reasoning: str = Field(description=f"A brief reasoning of how annotator Andy might label the post when assessing it.")
    Label: str = Field(description=f"An integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotator_label)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotator_label"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'ID': df['ID'].iloc[i],
            'Text': df['Text'].iloc[i],
            'Hate': df['Hate'].iloc[i],
            'Reasoning_LLM': output['Reasoning'],
            'Hate_LLM': output['Label'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(ghc_dis_a0_test)
output = get_output(prompts, ghc_dis_a0_test, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'ID': post_value['ID'],
        'Text': post_value['Text'],
        'Hate': post_value['Hate'],
        'Reasoning_LLM': post_value['Reasoning_LLM'],
        'Hate_LLM': post_value['Hate_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'ID',
    'Text',
    'Hate',
    'Hate_LLM',
    'Reasoning_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

df['Hate_LLM'] = df['Hate_LLM'].apply(lambda x: int(x))
df['Hate_diff'] = abs(df['Hate'] - df['Hate_LLM'])

# df['Hate_diff'].describe()
df.to_csv('03_annotator_modeling_a0.csv')

"""#### A11"""

### Annotator_11

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a helpful research assistant with expertise in detecting hate speech from social media posts.

        The task is to evaluate social media posts and predict the labels assigned by human annotator Andy, determining whether a post constitutes hate speech based on specific aspects of hate speech it may exhibit. Samples labeled by Andy are provided for reference. Carefully review these samples and consider how Andy would label the target post, and provide the label for it.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each target post, first provide a brief reasoning on how Andy would label it, based on specific aspects of hate speech it may exhibit.
        - Then, output an integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.

        Sample posts annotated by Andy:
        Post: The ONLY strong and clear message coming from the American 'intelligence' community, which @CBS (emphasis on BS) won't cover is 'we are a rogue state within a state and elected American officials can't reign us in' | Label: 0
        Post: Mexico is aiding and abetting, you stupid fucking dolt. Fuck you and fuck Mexico | Label: 1
        Post: After Sentencing, Cosby Camp Issues Disgusting Comment About Jesus https://www.westernjournal.com/ct/sentencing-cosby-camp-issues-disgusting-comment-jesus/?utm_source=Twitter&utm_medium=PostBottomSharingButtons&utm_content=2018-09-26&utm_campaign=websitesharingbuttons | Label: 0
        Post: 3) If you are picking up leftist vibes, it's because I used to be one of the zombies. Thankfully, I realized what I was doing to myself and others and smartened up after Bernie got screwed. I was a Bernie to Trump person (they DO exist). Libertarianism is the answer. Nationalism is the answer. | Label: 1
        Post: San Diego Union-Tribune staff evacuated after suspicious package found https://thehill.com/homenews/media/412937-san-diego-union-tribune-staff-evacuated-after-suspicious-package-found?amp&__twitter_impression=true | Label: 0
        Post: "t's not a religion.    It's a tribalistic, supremacist race with an insular culture that relies on dividing its host on order to survive.  "  What you are describing is a the filthy despicable JEW PIGS!  Jews are subversive LIARS and whatever they accuse others of they are guilty!  Jews are parasitical cancer and have been expelled from 1030 countries for a REASON!  Do you still belive the HOLOHOAX like this RETARD JAY? Jews are SCUM and you are a brainwashed RETARD! | Label: 1
        Post: How about Satanism, the religion of the Great Liar himself.! | Label: 0
        Post: https://youtu.be/Lm_3sJ8CRuU  | Label: 1
        Post: Miscegenation -- like single mommery, extortionist child support, and volcuckoldry -- is a revolt against paternity certainty. Mudsharking severs the aesthetic connection between parent and child, reducing the resemblance of child to father and increasing uncertainty of the child's paternal provenance.  | Label: 1
        Post: Senate Intel Committee Delivers Interview Invite to Assange!!! https://www.youtube.com/watch?v=FXjyaVeFQ08&t=383s | Label: 0 """

        p_user = f"""Here is the target post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotator_label(BaseModel):

    """Detect Annotator Andy's label on Hate Speech Posts"""
    Reasoning: str = Field(description=f"A brief reasoning of how annotator Andy might label the post when assessing it.")
    Label: str = Field(description=f"An integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotator_label)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotator_label"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'ID': df['ID'].iloc[i],
            'Text': df['Text'].iloc[i],
            'Hate': df['Hate'].iloc[i],
            'Reasoning_LLM': output['Reasoning'],
            'Hate_LLM': output['Label'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(ghc_dis_a11_test)
output = get_output(prompts, ghc_dis_a11_test, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'ID': post_value['ID'],
        'Text': post_value['Text'],
        'Hate': post_value['Hate'],
        'Reasoning_LLM': post_value['Reasoning_LLM'],
        'Hate_LLM': post_value['Hate_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'ID',
    'Text',
    'Hate',
    'Hate_LLM',
    'Reasoning_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

df['Hate_LLM'] = df['Hate_LLM'].apply(lambda x: int(x))
df['Hate_diff'] = abs(df['Hate'] - df['Hate_LLM'])

"""#### A0_dis"""

dis_a0_pos = dis[dis['Annotator_0'] == 1].sample(n=24, random_state=42)
dis_a0_pos_10 = dis_a0_pos.head(10)
dis_a0_pos_14 = dis_a0_pos.tail(14)

dis_a11_pos = dis[dis['Annotator_11'] == 1].sample(n=19, random_state=42)
dis_a11_pos = dis_a11_pos.sample(frac=1, random_state=42)

dis_a0_dev = pd.concat([dis_a0_pos_10.head(), dis_a11_pos.head()]).sample(frac=1, random_state=42).reset_index(drop=True)
dis_a11_dev = pd.concat([dis_a0_pos_10.tail(), dis_a11_pos.head()]).sample(frac=1, random_state=42).reset_index(drop=True)

dis_test = pd.concat([dis_a0_pos_14, dis_a11_pos.tail(14)]).sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize an empty string to store the merged paragraph
paragraph_a0, paragraph_a11 = "", ""

# Iterate through each row in the dataframe
for index, row in dis_a0_dev.iterrows():
    post = row['Text']
    label = row['Annotator_0']
    paragraph_a0 += f" Post: {post} | Label: {label} \n"

print(paragraph_a0)

for index, row in dis_a11_dev.iterrows():
    post = row['Text']
    label = row['Annotator_11']
    paragraph_a11 += f" Post: {post} | Label: {label} \n"

print(paragraph_a11)

### Annotator_0

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a helpful research assistant with expertise in detecting hate speech from social media posts.

        The task is to evaluate social media posts and predict the labels assigned by human annotator Andy, determining whether a post constitutes hate speech based on specific aspects of hate speech it may exhibit. Samples labeled by Andy are provided for reference. Carefully review these samples and consider how Andy would label the target post, and provide the label for it.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each target post, first provide a brief reasoning on how Andy would label it, based on specific aspects of hate speech it may exhibit.
        - Then, output an integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.

        Sample posts annotated by Andy:
        Post: Granny Schicklgruber's Cherries Jewbilee Recipe  | Label: 0
        Post: Oh, but TWITter is a fun place today. I feel the shadow of ban coming upon my sunny social media day. | Label: 1
        Post: Carbs, which make you tired so you don't exercise, and hungry so you eat more. They think they are eating the right things because the USDA tells them so. But they probably don't really eat much more than thinner people. Their diet reduces their metabolism so much that 2000 calories a day can keep them fat.  Eventually it affects mood and cognition, and viola, you have a progressive feminist. Fat, infertile, with perverse desires, and distorted ideas. | Label: 0
        Post: OH ..I WANT AN INVESTIGATION ON THIS!!!!....TO SEE HOW FAST IT SPIT AND KICK THEM IN THEIR DEMON RATS FACES!! | Label: 1
        Post: Britbongs, I have a question for you. Would it be a hate crime for you to spread out a blanket and eat a yummy pork sandwich in front of a bunch of muzzies doing their prayers to the meteorite? | Label: 0
        Post: And we don't give two fucks what you pretend to think in attempt to silence us. | Label: 1
        Post: Doolittle also had them squirming when he basically said nothing but violence would make any difference.  They were scrambling when big brained nibba blasted them with the truth. | Label: 0
        Post: He's  a professional  ?? Come on, my parrot cries more believable.  And the parrot can bark ! Woof Shep, go back to your doghouse& out of the rain | Label: 1
        Post: Sadiq Khan GO TO HELL  | Label: 1
        Post: Race mixing once done...it can't be undone | Label: 0  """

        p_user = f"""Here is the target post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotator_label(BaseModel):

    """Detect Annotator Andy's label on Hate Speech Posts"""
    Reasoning: str = Field(description=f"A brief reasoning of how annotator Andy might label the post when assessing it.")
    Label: str = Field(description=f"An integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotator_label)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotator_label"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'Text': df['Text'].iloc[i],
            'Hate': df['Annotator_0'].iloc[i],
            'Reasoning_LLM': output['Reasoning'],
            'Hate_LLM': output['Label'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(dis_test)
output = get_output(prompts, dis_test, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'Text': post_value['Text'],
        'Hate': post_value['Hate'],
        'Reasoning_LLM': post_value['Reasoning_LLM'],
        'Hate_LLM': post_value['Hate_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'Text',
    'Hate',
    'Hate_LLM',
    'Reasoning_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

df['Hate_LLM'] = df['Hate_LLM'].apply(lambda x: int(x))
df['Hate_diff'] = abs(df['Hate'] - df['Hate_LLM'])

"""#### A11_dis"""

### Annotator_11

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a helpful research assistant with expertise in detecting hate speech from social media posts.

        The task is to evaluate social media posts and predict the labels assigned by human annotator Andy, determining whether a post constitutes hate speech based on specific aspects of hate speech it may exhibit. Samples labeled by Andy are provided for reference. Carefully review these samples and consider how Andy would label the target post, and provide the label for it.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each target post, first provide a brief reasoning on how Andy would label it, based on specific aspects of hate speech it may exhibit.
        - Then, output an integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.

        Sample posts annotated by Andy:
        Post: Granny Schicklgruber's Cherries Jewbilee Recipe  | Label: 1
        Post: I'm afraid Hillary Clinton won't make it to trial at the rate she continues to harm herself. http://www.dailymail.co.uk/news/article-5506823/Hillary-Clinton-fractures-wrist-slipping-bathtub.html #HelloMyNameIsHillaryAndImAnAlcoholic | Label: 0
        Post: Carbs, which make you tired so you don't exercise, and hungry so you eat more. They think they are eating the right things because the USDA tells them so. But they probably don't really eat much more than thinner people. Their diet reduces their metabolism so much that 2000 calories a day can keep them fat.  Eventually it affects mood and cognition, and viola, you have a progressive feminist. Fat, infertile, with perverse desires, and distorted ideas. | Label: 1
        Post: What does that have to do with the "context" of this verse? Wake up! The Bible is a forgery written by Jewish priests to promote their power! | Label: 0
        Post: Britbongs, I have a question for you. Would it be a hate crime for you to spread out a blanket and eat a yummy pork sandwich in front of a bunch of muzzies doing their prayers to the meteorite? | Label: 1
        Post: A soyboy C U C K explains how he was raped by his wife’s boyfriend, so he’s sending her to go meet with him one last time to “break things off.” | Label: 0
        Post: Doolittle also had them squirming when he basically said nothing but violence would make any difference.  They were scrambling when big brained nibba blasted them with the truth. | Label: 1
        Post: 12 Oct 2018 - This Lady is Koo-Koo... Loco in la cabeza - Senate Democratic Candidate: “I Don’t Care” if People Go Fight for the Taliban Against US (AUDIO) - https://www.thegatewaypundit.com/2018/10/senate-democratic-candidate-i-dont-care-if-people-go-fight-for-the-taliban-against-us-video/ | Label: 0
        Post: Apparently some audience members don’t appreciate obvious lies.  | Label: 0
        Post: Race mixing once done...it can't be undone | Label: 1   """

        p_user = f"""Here is the target post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotator_label(BaseModel):

    """Detect Annotator Andy's label on Hate Speech Posts"""
    Reasoning: str = Field(description=f"A brief reasoning of how annotator Andy might label the post when assessing it.")
    Label: str = Field(description=f"An integer label for the target post: 1 if Andy thinks it entails hate speech, 0 if not.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotator_label)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotator_label"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'Text': df['Text'].iloc[i],
            'Hate': df['Annotator_11'].iloc[i],
            'Reasoning_LLM': output['Reasoning'],
            'Hate_LLM': output['Label'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(dis_test)
output = get_output(prompts, dis_test, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'Text': post_value['Text'],
        'Hate': post_value['Hate'],
        'Reasoning_LLM': post_value['Reasoning_LLM'],
        'Hate_LLM': post_value['Hate_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'Text',
    'Hate',
    'Hate_LLM',
    'Reasoning_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

df['Hate_LLM'] = df['Hate_LLM'].apply(lambda x: int(x))
df['Hate_diff'] = abs(df['Hate'] - df['Hate_LLM'])

"""### S4 Joint Annotator Modeling

"""

ghc_dis = ghc[ghc['Annotator'].isin([0, 11, 13])]

# Count the occurrences of each 'ID'
id_counts = ghc_dis['ID'].value_counts()

# Filter IDs that appear exactly three times
ids_to_keep = id_counts[id_counts == 3].index

# Keep only rows where 'ID' is in the list of IDs that appear three times
ghc_dis = ghc_dis[ghc_dis['ID'].isin(ids_to_keep)]

# Pivot the dataframe to create the desired structure
ghc_dis = ghc_dis.pivot(index=['ID', 'Text'], columns='Annotator', values='Hate')

# Rename the columns to match the desired format
ghc_dis.columns = [f'Hate_{int(col)}' for col in ghc_dis.columns]

# Reset the index to turn it back into a regular dataframe
ghc_dis = ghc_dis.reset_index()

ghc_dis['Hate_MV'] = ghc_dis[['Hate_0', 'Hate_11', 'Hate_13']].mode(axis=1)[0]
ghc_dis['Hate_MV'] = ghc_dis['Hate_MV'].apply(lambda x: int(x))

# Sample n rows where 'Hate_MV' is 1
hate_speech_sample = ghc_dis[ghc_dis['Hate_MV'] == 1].sample(n=56, random_state=42)
hate_speech_sample = hate_speech_sample.sample(frac=1, random_state=42)

# Sample n rows where 'Hate_MV' is 0
non_hate_speech_sample = ghc_dis[ghc_dis['Hate_MV'] == 0].sample(n=56, random_state=42)
non_hate_speech_sample = non_hate_speech_sample.sample(frac=1, random_state=42)

# Combine the two samples
ghc_dis_dev = pd.concat([hate_speech_sample.head(6), non_hate_speech_sample.head(6)])
ghc_dis_test = pd.concat([hate_speech_sample.tail(50), non_hate_speech_sample.tail(50)])

# Shuffle the combined sample to mix hate and non-hate rows
ghc_dis_dev = ghc_dis_dev.sample(frac=1, random_state=42).reset_index(drop=True)
ghc_dis_test = ghc_dis_test.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize an empty string to store the merged paragraph
paragraph = ""

# Iterate through each row in the dataframe
for index, row in ghc_dis_dev.iterrows():
    post = row['Text']
    label_a = row['Hate_0']
    label_b = row['Hate_11']
    label_c = row['Hate_13']
    paragraph += f" Post: {post} | Label by Andy: {label_a} | Label by Ben: {label_b} | Label by Chris: {label_c} \n"

print(paragraph)

model = 'gpt-4o'

def get_prompts(df):

    posts = df['Text'].tolist()

    prompts = []
    for post in posts:
        p_system = f"""You are a helpful research assistant with expertise in detecting hate speech from social media posts.

        Objective: The task is to evaluate social media posts and predict the labels assigned by human annotators Andy, Ben, and Chris, determining whether a post constitutes hate speech based on specific aspects of hate speech it may exhibit. Samples labeled by them are provided for reference. Carefully review these samples and consider how they would label the target post, respectively, and provide a label for each of them.

        Aspects of Hate Speech to Consider:
        - Assaults on Human Dignity: Does the post demean or degrade individuals or groups based on race, ethnicity, gender, religion, sexual orientation, or other protected characteristics?
        - Calls for Violence: Does the post incite or encourage physical harm or violence against individuals or groups?
        - Vulgarity and/or Offensive Language: Does the post contain profanity, slurs, or other offensive language that may or may not be directed at individuals or groups?

        Instructions:
        - Review the Post: Carefully read the social media post and consider its content in relation to the three aspects of hate speech listed above.
        - Assess Agreement: 1) High Disagreement: If the post contains ambiguous language or context, or if it is likely to be interpreted differently based on subjective views about the severity or nature of hate speech, categorize the post as 'high' disagreement. 2) Low Disagreement: If the post clearly falls into one or more of the hate speech categories or is obviously not hate speech, categorize the post as 'low' disagreement due to likely annotator consensus.
        - Context Sensitivity: Consider the cultural, social, and linguistic context that may influence the interpretation of the post. Posts that rely heavily on nuanced or context-specific language may result in higher disagreement.

        Output Format:
        - For each target post, first provide a brief reasoning on how Andy, Ben, and Chris would label it, based on specific aspects of hate speech it may exhibit.
        - Then, output for Andy, Ben, and Chris an integer label for the target post, respectively: 1 if the annotator thinks it entails hate speech, 0 if not. E.g. Label by Andy: 0 | Label by Ben: 1 | Label by Chris: 0.

        Sample posts annotated by Andy, Ben, and Chris:
        Post: This should be noted. India is deploying motivational speakers to ignite passion into its demoralized army. This must be taken seriously. This is psychological preparation for something huge. Signs of a coming war? Hmmmm | Label by Andy: 1 | Label by Ben: 0 | Label by Chris: 0
        Post: Oh, but TWITter is a fun place today. I feel the shadow of ban coming upon my sunny social media day. | Label by Andy: 1 | Label by Ben: 0 | Label by Chris: 0
        Post: I don't mind the occasional gas the kikes thrown in as long as it's satirical and makes me laugh a bit at their discomfort. 😎 | Label by Andy: 1 | Label by Ben: 1 | Label by Chris: 1
        Post: It's 100% OK - thank you 😉 | Label by Andy: 0 | Label by Ben: 0 | Label by Chris: 0
        Post: What every good little faggot needs. | Label by Andy: 1 | Label by Ben: 0 | Label by Chris: 1
        Post: How many MILLIONS of ILLEGAL ALIENS voted for HILLARY⁉️Who cares, that had no IMPACT on the election. RETARDS. | Label by Andy: 1 | Label by Ben: 1 | Label by Chris: 0
        Post: Feminism is now Female Supremacism as women have all the rights that men have and they still have special privileges in the courts and in our culture. Cultural Marxists who hate America, Whites, and the West are the leaders and preachers of Third Wave Feminism. | Label by Andy: 1 | Label by Ben: 1 | Label by Chris: 0
        Post: Communism and socialism is going to rock Europe this summer, all these tiny "Rules" have also been implimented in California USA... Europe is heading for war. | Label by Andy: 1 | Label by Ben: 0 | Label by Chris: 0
        Post: kikes are liars and thieves. | Label by Andy: 1 | Label by Ben: 1 | Label by Chris: 0
        Post: "All our dignity lies in our thoughts." - Blaise Pascal.  | Label by Andy: 0 | Label by Ben: 0 | Label by Chris: 0
        Post: If we can get Nazis away from Nazism, especially away from anti-Semitism, that is better. Nazism and anti-Semitism are the problem, not ‘true racism.’ Do not be a typical leftist liberal Antifa-type and paint ‘true racists’ and Nazis with the same brush. It is reverse psychology. | Label by Andy: 0 | Label by Ben: 1 | Label by Chris: 1
        Post: I feel your pain.  | Label by Andy: 1 | Label by Ben: 0 | Label by Chris: 0 """

        p_user = f"""Here is the target post: {post}"""

        prompts.append([p_system, p_user])

    return prompts

class detect_annotator_label(BaseModel):

    """Detect Annotators' label on Hate Speech Posts"""
    Reasoning: str = Field(description=f"A brief reasoning of how annotator Andy might label the post when assessing it.")
    Label_Andy: str = Field(description=f"An integer label for the target post by Andy: 1 if the annotator thinks it entails hate speech, 0 if not.")
    Label_Ben: str = Field(description=f"An integer label for the target post by Ben: 1 if the annotator thinks it entails hate speech, 0 if not.")
    Label_Chris: str = Field(description=f"An integer label for the target post by Chris: 1 if the annotator thinks it entails hate speech, 0 if not.")
    Raw_output: str = Field(description=f"The raw text output of the LLM.")

functions = [
    convert_to_openai_function(detect_annotator_label)
]

def get_output(prompts, df, model):

    posts = {}

    # define model
    model = ChatOpenAI(
          model=model,
          openai_api_key=openai.api_key,
          temperature=0,
          )
    model = model.bind(
        functions=functions,
        function_call={"name":"detect_annotator_label"},
        )

    # define parser
    parser = JsonOutputFunctionsParser()

    for i in range(len(prompts)):

        p_system, p_user = [*prompts[i]]

        # define prompt message
        prompt = ChatPromptTemplate.from_messages([
            ("system", p_system),
            ("user", "{input}")
            ])

        # create a chatbot chain
        chain = prompt | model | parser

        input = p_user
        output = chain.invoke({"input": input})

        post_id = 'post_' + str(i)
        posts[post_id] = {
            'ID': df['Text'].iloc[i],
            'Text': df['Text'].iloc[i],
            'Hate_MV': df['Hate_MV'].iloc[i],
            'Hate_0': df['Hate_0'].iloc[i],
            'Hate_11': df['Hate_11'].iloc[i],
            'Hate_13': df['Hate_13'].iloc[i],
            'Reasoning_LLM': output['Reasoning'],
            'Hate_0_LLM': output['Label_Andy'],
            'Hate_11_LLM': output['Label_Ben'],
            'Hate_13_LLM': output['Label_Chris'],
            'Raw_output_LLM': output['Raw_output'],
            }

    return posts

prompts = get_prompts(ghc_dis_test)
output = get_output(prompts, ghc_dis_test, model=model)

# Extracting data for DataFrame
data = []
for post_key, post_value in output.items():
    row = {
        'ID': post_value['ID'],
        'Text': post_value['Text'],
        'Hate_MV': post_value['Hate_MV'],
        'Hate_0': post_value['Hate_0'],
        'Hate_11': post_value['Hate_11'],
        'Hate_13': post_value['Hate_13'],
        'Reasoning_LLM': post_value['Reasoning_LLM'],
        'Hate_0_LLM': post_value['Hate_0_LLM'],
        'Hate_11_LLM': post_value['Hate_11_LLM'],
        'Hate_13_LLM': post_value['Hate_13_LLM'],
        'Raw_output_LLM': post_value['Raw_output_LLM'],
    }
    data.append(row)

# Creating DataFrame
columns = [
    'Text',
    'Hate_MV',
    'Hate_0',
    'Hate_11',
    'Hate_13',
    'Reasoning_LLM',
    'Hate_0_LLM',
    'Hate_11_LLM',
    'Hate_13_LLM',
    'Raw_output_LLM',
    ]
df = pd.DataFrame(data, columns=columns)

df['Hate_MV_LLM'] = df[['Hate_0_LLM', 'Hate_11_LLM', 'Hate_13_LLM']].mode(axis=1)[0]

df['Hate_MV_LLM'] = df['Hate_MV_LLM'].apply(lambda x: int(x))
df['Hate_0_LLM'] = df['Hate_0_LLM'].apply(lambda x: int(x))
df['Hate_11_LLM'] = df['Hate_11_LLM'].apply(lambda x: int(x))
df['Hate_13_LLM'] = df['Hate_13_LLM'].apply(lambda x: int(x))

df['Hate_MV_diff'] = abs(df['Hate_MV'] - df['Hate_MV_LLM'])
df['Hate_0_diff'] = abs(df['Hate_0'] - df['Hate_0_LLM'])
df['Hate_11_diff'] = abs(df['Hate_11'] - df['Hate_11_LLM'])
df['Hate_13_diff'] = abs(df['Hate_13'] - df['Hate_13_LLM'])
