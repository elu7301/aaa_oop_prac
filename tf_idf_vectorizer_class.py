from count_vectorizer_class import CountVectorizer
from tf_idf_transformer_class import TfidfTransformer


class TfidfVectorizer(CountVectorizer):
    """
    Text vectorizer that converts an iterable object
    into a matrix of TF-IDF values.

    Inherits from CountVectorizer.

    Methods:
        fit_transform(docs) : Fit the vectorizer on the documents
            and transform the documents into a matrix of TF-IDF values.
    """

    def __init__(self) -> None:
        """
        Initialize TfidfVectorizer.
        """
        super().__init__()
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf = []

    def fit_transform(self, docs: list | tuple | set) -> list:
        """
        Fit the vectorizer on the documents
        and transform the documents into a matrix of TF-IDF values.

        Args:
            docs : List of documents.

        Returns:
            tfidf : Matrix of TF-IDF values.
        """
        count_matrix = super().fit_transform(docs)
        self.tfidf = self.tfidf_transformer.fit_transform(count_matrix)

        return self.tfidf


if __name__ == '__main__':
    # Test 1
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
