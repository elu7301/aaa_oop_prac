from tf_function import tf_transform
from idf_function import idf_transform


class TfidfTransformer:
    """
    Class for computing TF-IDF transformation on a count matrix.

    Methods:
        fit_transform(count_matrix) : Compute the TF-IDF
            transformation on the given count matrix.
    """

    def __init__(self):
        """
        Initialize TfidfTransformer.
        """
        self.tf = []
        self.idf = []
        self.tfidf = []

    def fit_transform(self, count_matrix: list) -> list:
        """
        Compute the TF-IDF transformation on the given count matrix.

        Args:
            count_matrix : The count matrix
                representing the document-term frequencies.

        Returns:
            tf_idf_matrix : The TF-IDF matrix, where
                each value is the TF-IDF score for a term in a document.
        """
        docs_num = len(count_matrix)
        words_num = len(count_matrix[0])

        self.tf = tf_transform(count_matrix)
        self.idf = idf_transform(count_matrix)

        tf_idf_matrix = [[round(self.idf[j] * self.tf[i][j], 3) for j
                          in range(words_num)] for i in range(docs_num)]
        self.tfidf = tf_idf_matrix

        return self.tfidf


if __name__ == '__main__':
    # Test 1
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]

    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)

    print(tfidf_matrix)
