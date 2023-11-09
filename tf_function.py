def tf_transform(count_matrix: list) -> list:
    """
    Compute the term frequency (TF) transformation
    on the given count matrix.

    Args:
        count_matrix : The count matrix
            representing the document-term frequencies.

    Returns:
        new_matrix : The TF matrix, where
            each value is the TF score for a term in a document.
    """
    new_matrix = []

    for vector in count_matrix:
        new_matrix.append([round(elem / sum(vector), 3) for elem in vector])

    return new_matrix


if __name__ == '__main__':
    # Test 1
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]

    tf_matrix = tf_transform(count_matrix)

    print(tf_matrix)
