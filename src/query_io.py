import csv
import os
from collections import defaultdict

import pandas as pd


def processTest(result_path):
    test_files = os.listdir(result_path)
    results = defaultdict(list)

    for file in test_files:
        query_id = int(file.split(".")[0])
        results[query_id] = []

        result_df = pd.read_csv(
            os.path.join(result_path, file),
            sep=r"\s+",
            header=None,
            names=["QueryID", "DocID", "Rating"],
            engine="python",
        )
        result_df = result_df.dropna(subset=["DocID", "Rating"])

        for _, row in result_df.iterrows():
            rating = row["Rating"]
            if pd.isna(rating):
                continue
            if int(rating) != -1:
                results[query_id].append(int(row["DocID"]))

    return results


def loadQueries(query_file):
    queries = defaultdict(dict)
    with open(query_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            query_id = int(row[0])
            queries[query_id] = row[1]
    return queries
