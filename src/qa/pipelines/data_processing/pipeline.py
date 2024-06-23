"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calculate_tf_idf_matrix, clean_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_dataframe,
                inputs=["train", "params:col_titles"],
                outputs="cleaned_data_titles",
                name="clean_text_node_titles",
            ),
            node(
                func=clean_dataframe,
                inputs=["train", "params:col_abstracts"],
                outputs="cleaned_data_abstracts",
                name="clean_text_node_abstracts",
            ),
            node(
                func=calculate_tf_idf_matrix,
                inputs=["cleaned_data_abstracts", "params:col_titles"],
                outputs=["X_tfidf_titles", "vectorizer_titles"],
                name="tf_idf_titles",
            ),
            node(
                func=calculate_tf_idf_matrix,
                inputs=["cleaned_data_abstracts", "params:col_abstracts"],
                outputs=["X_tfidf_abstracts", "vectorizer_abstract"],
                name="tf_idf_abstract",
            ),
        ]
    )
