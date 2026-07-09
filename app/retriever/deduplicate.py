def deduplicate(docs):
    unique = {}
    ordered = []

    for doc in docs:

        doc_id = doc.metadata["doc_id"]

        if doc_id not in unique:
            unique[doc_id] = True
            ordered.append(doc)

    return ordered