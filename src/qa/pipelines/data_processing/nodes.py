"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = stopwords.words("english")


lematizer = WordNetLemmatizer()


def clean_text(text):
    """ "
    This function receives a text and returns it cleaned.

    Args:
        text (str): a text to be cleaned

    Returns:
        str: the cleaned text
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    tokens = [lematizer.lemmatize(word) for word in tokens]
    text = " ".join(tokens)
    text = text.strip()
    return text


def clean_dataframe(df: pd.DataFrame, col_target: str):
    """
    This function receives a DataFrame and a column name and returns the DataFrame with the column cleaned.

    Args:
        df (pd.DataFrame): a DataFrame to be transformed
        col_target (str): the column name to be cleaned

    Returns:
        pd.DataFrame: the DataFrame with the column cleaned
    """
    df[col_target] = df[col_target].apply(clean_text)
    return df


def calculate_tf_idf_matrix(df: pd.DataFrame, col_target: str):
    """
    This function receives a DataFrame and a column name and returns the TF-IDF matrix.

    Args:
        df (pd.DataFrame): a DataFrame to be transformed
        col_target (str): the column name to be used

    Returns:
        matrix: the TF-IDF matrix
        vectorizer: the vectorizer used to transform the matrix
    """
    vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.005)
    X = vectorizer.fit_transform(df[col_target])
    X = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return X, vectorizer


def nmf_decomposition(X, n_components, vectorizer):
    """
    This function receives a matrix X and returns the decomposition based on NMF.

    Args:
        X (matrix): a matrix to be decomposed
        n_components (int): the number of components to be used in the decomposition
        vectorizer: the vectorizer used to transform the matrix

    Returns:
        pd.DataFrame: the elements matrix
        pd.DataFrame: the samples matrix
    """
    nmf = NMF(n_components=n_components)
    W = nmf.fit_transform(X)
    df_elements = pd.DataFrame(
        nmf.components_, columns=vectorizer.get_feature_names_out()
    )
    df_samples = pd.DataFrame(W)
    return df_elements, df_samples
