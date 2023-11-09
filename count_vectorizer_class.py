class CountVectorizer:
    """
    Text vectorizer that converts
    an iterable object into a matrix of token counts.

    Args:
        lowercase : Convert text to lowercase. Default is True.

    Methods:
        fit_transform(docs): Fit the vectorizer on the documents
            and transform the documents into a matrix of token counts.
        get_feature_names(): Get the list of unique tokens (features)
            in the order of their appearance in the count matrix.
    """

    def __init__(self, lowercase: bool = True) -> None:
        """
        Initialize CountVectorizer.

        Args:
            lowercase : Convert text to lowercase. Default is True.
        """
        self.lowercase = lowercase
        self.vector = {}
        self.matrix = []

    def fit_transform(self, docs: tuple | set | list) -> list:
        """
        Fit the vectorizer on the documents
        and transform the documents into a matrix of token counts.

        Args:
            docs : List of documents.

        Returns:
            matrix : Matrix of token counts.
        """
        if self.lowercase:
            docs = list(map(str.lower, docs))

        temp = []

        for doc in docs:
            temp.extend(doc.split())
        counter = dict.fromkeys(temp)

        self.vector = dict.fromkeys(counter, 0)
        for doc in docs:
            temp_vector = self.vector.copy()
            for word in doc.split():
                temp_vector[word] += 1
            self.matrix.append(list(temp_vector.values()))

        return self.matrix

    def get_feature_names(self) -> list:
        """
        Get the list of unique tokens (features)
        in the order of their appearance in the count matrix.

        Returns:
            feature_names : List of unique tokens (features).
        """
        return list(self.vector.keys())


if __name__ == '__main__':
    # Test 1
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(count_matrix)
