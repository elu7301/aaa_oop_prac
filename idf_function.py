import math


def idf_transform(count_matrix: list) -> list:
    """
    Compute the inverse document frequency (IDF)
    transformation on the given count matrix.

    Args:
        count_matrix : The count matrix
            representing the document-term frequencies.

    Returns:
        new_vector : The IDF vector, where
            each value is the IDF score for a term.
    """
    docs_num = len(count_matrix)
    words_num = len(count_matrix[0])
    new_vector = [0 for _ in range(words_num)]

    for vector in count_matrix:
        for i in range(words_num):
            new_vector[i] += bool(vector[i])

    my_func = lambda x: round(math.log((docs_num + 1) / (x + 1)) + 1, 3)
    new_vector = list(map(my_func, new_vector))

    return new_vector


if __name__ == '__main__':
    # Test 1
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]

    idf_vector = idf_transform(count_matrix)

    print(idf_vector)
