"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calculate_tf_idf_matrix, clean_dataframe, nmf_decomposition


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
                inputs=["cleaned_data_titles", "params:col_abstracts"],
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
            node(
                func=nmf_decomposition,
                inputs=["X_tfidf_titles", "params:n_titles", "vectorizer_titles", "params:type_titles"],
                outputs=["F_titles", "K_titles"],
                name="nmf_titles",
            ),
            node(
                func=nmf_decomposition,
                inputs=["X_tfidf_abstracts", "params:n_abstracts", "vectorizer_abstract", "params:type_abstracts"],
                outputs=["F_abstracts", "K_abstracts"],
                name="nmf_abstracts",
            ),
            node(inputs=["F_titles"],outputs="results_titles",  name  =   "predictions_titles"),
            node(inputs=["F_abstracts"],outputs="results_abstracts", name   =  "predictions_abstracts")
        ]
    )
