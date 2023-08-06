import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize


def nlp_random_search_modeler(model, vectorizer, cross_validation=5, vectorizer_params=None, model_params=None, random_state=42, num_iter=20):
   
    """
    Perform a randomized search for hyperparameter tuning in NLP models.

    Parameters:
    -----------
    model : estimator
        The NLP model to be used, which can be any scikit-learn classifier.

    vectorizer : transformer
        The vectorizer responsible for converting text data into numerical features, such as CountVectorizer or TfidfVectorizer.

    cross_validation : cross-validation generator or iterable, default=5
        Determines the cross-validation strategy, specifying how the data should be split for evaluation.

    vectorizer_params : dict, optional
        Dictionary of parameters to be passed to the vectorizer. These parameters control aspects such as preprocessing, feature extraction, and 
        dimensionality reduction.

    model_params : dict, optional
        Dictionary of parameters to be passed to the model. These parameters determine the specific configuration and behavior of the chosen NLP model.

    random_state : int, default=42
        The random seed used for reproducibility of results during the random search.

    Returns:
    --------
    rs : RandomizedSearchCV object
        The resulting RandomizedSearchCV object that can be used for fitting and evaluating the model.

    """
    pipe = Pipeline([
        ('vec', vectorizer),
        ('model', model)
    ])

    pgrids = {}

    if vectorizer_params:
        for key, val in vectorizer_params.items():
            pgrids[f"vec__{key}"] = val

    if model_params:
        for key, val in model_params.items():
            pgrids[f"model__{key}"] = val

    rs = RandomizedSearchCV(pipe, param_distributions=pgrids, cv=cross_validation, random_state=random_state, n_iter=num_iter)
    return rs

##################################

# Function for lemmatizing
def lemmatize_text(text):
    """
    Lemmatizes the words in a given text.

    Parameters
    ----------
    text : str
        The input text to be lemmatized.

    Returns
    -------
    str
        The lemmatized text.

    """
    # Split the text into individual words
    split_text = text.split(' ')

    # Instantiate a WordNetLemmatizer object
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word in the text and rejoin them
    return ' '.join([lemmatizer.lemmatize(word) for word in split_text])


##################################

# Function for stemming
def stem_text(text):
    """
    Stems the words in a given text.

    Parameters
    ----------
    text : str
        The input text to be stemmed.

    Returns
    -------
    str
        The stemmed text.

    """
    # Split the text into individual words
    split_text = text.split(' ')

    # Instantiate a PorterStemmer object
    p_stemmer = PorterStemmer()

    # Stem each word in the text and rejoin them
    return ' '.join([p_stemmer.stem(word) for word in split_text])

######################################

## The following two functions are used for lemmatizing based on part of speech
def custom_lemmatize(word, tag):
    """
    Lemmatizes a word based on its part-of-speech (POS) tag.

    Parameters
    ----------
    word : str
        The word to be lemmatized.
    tag : str
        The POS tag of the word.

    Returns
    -------
    str
        The lemmatized word.

    """
    # Mapping of POS tags to WordNet POS tags
    mapper = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }

    # Get the corresponding WordNet POS tag
    pos = mapper.get(tag[0])

    # Lemmatize the word using the WordNetLemmatizer if a valid POS tag is found, otherwise return the word as is
    return WordNetLemmatizer().lemmatize(word, pos) if pos else word

#################

def pos_lemmatizer(text):
    """
    Lemmatizes the words in a given text based on their part-of-speech (POS) tags.

    Parameters
    ----------
    post : str
        The input text to be lemmatized.

    Returns
    -------
    str
        The lemmatized text.

    """
    # Split the post into individual words
    text_list = text.split(' ')

    # Use nltk.pos_tag to assign POS tags to each word
    pos_tags = nltk.pos_tag(text_list)

    # Lemmatize each word based on its POS tag using the custom_lemmatize function, and rejoin them
    return ' '.join([custom_lemmatize(token, tag) for token, tag in pos_tags])

###############################
###################################

def text_feature_engineering(data, column, prefix=None):
    
    """
    Perform text feature engineering on a DataFrame column.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    column : str
        The column in the DataFrame to perform feature engineering on.
    prefix : str, optional
        Prefix to add to the new column names after feature engineering,
        by default None.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with added text analysis features.

    """

    # Calculate the length of each text
    data['char_length'] = data[column].apply(len)

    # Calculate the number of words in each text
    data['word_count'] = data[column].apply(lambda x: len(str(x).split()))

    # Calculate the mean length of words in each text
    data['mean_word_length'] = data[column].apply(lambda x: len(''.join(x.split())) / len(x.split()))

    # Calculate the number of sentences in each text
    data['count_sentence'] = data[column].apply(lambda x: len(sent_tokenize(x)))

    # Calculate the mean number of words per sentence in each text
    data['mean_word_per_sentence'] = data[column].apply(lambda x: np.mean([len(sent.split()) for sent in sent_tokenize(x)]))

    # Calculate the mean number of characters per sentence in each text
    data['mean_char_per_sentence'] = data[column].apply(lambda x: np.mean([len(sent) for sent in sent_tokenize(x)]))

    if prefix:
        # Rename the columns with the provided prefix
        data.rename(columns={name: f"{name}_{prefix}" for name in data.columns[-6:]}, inplace=True)

    return data

########################

# Please note that this function is specifically designed for the purpose of collecting data from the Investing and Personal Finance subreddits. It is important # to keep in mind that modifications will be necessary if you intend to use it for other text cleaning tasks.

def clean_data(data):
    
    # Replace missing values with an empty string
    data['text'].replace(np.nan, '', inplace=True)
    
    # Combine title and text columns into a single 'full_text' column
    data['full_text'] = data['title'] + ' ' + data['text']
    
    # Convert 'full_text' to lowercase and remove subreddit names
    data['full_text'] = data['full_text'].str.lower().str.replace('investing', '')
    data['full_text'] = data['full_text'].str.lower().str.replace('personal finance', '')
    
    # Convert subreddit names to numerical values (1 for 'personalfinance', 0 for 'investing')
    data['subreddit'] = (data['subreddit'] == 'personalfinance') * 1
    
    # Keep only the 'subreddit' and 'full_text' columns
    data = data[['subreddit', 'full_text']]
    
    return data