import os
from collections import defaultdict


def loadDocuments(folder_path):
    documents = defaultdict(str)
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            docID = int(filename.split(".")[0])
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                documents[docID] = file.read()
    docIDs = list(documents.keys())
    return docIDs, documents
