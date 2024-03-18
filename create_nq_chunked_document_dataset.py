import argparse
import json
import math
import os

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from tqdm import tqdm


def split_document(document_tokens, chunk_size_in_tokens):
    chunks = []
    for i in range(math.ceil(len(document_tokens)/chunk_size_in_tokens)):
        start_index = i * chunk_size_in_tokens
        end_index = min((i + 1) * chunk_size_in_tokens, len(document_tokens))
        chunks.append(document_tokens[start_index:end_index])
    return chunks

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create Natural Questions Chunked Documents Dataset")
    parser.add_argument("--save-location", type=str, required=True)
    parser.add_argument("--chunk-size-in-tokens", type=int, default=128)
    args = parser.parse_args()

    train_titles = set()
    nq_train_dataset = load_dataset("natural_questions", split="train", cache_dir=args.save_location)
    chunked_nq_train_dataset = []
    for example in tqdm(nq_train_dataset):
        document = example["document"]
        if document["title"] not in train_titles:
            train_titles.add(document["title"])
            tokens = document["tokens"]
            tokens_filtered = [
                token for token, is_html in zip(tokens["token"], tokens["is_html"]) if not is_html
            ]
            document_tokens = document["title"].split(" ") + tokens_filtered # prepend title

            document_chunks = split_document(document_tokens, args.chunk_size_in_tokens)
            for document_chunk in document_chunks:
                chunked_example = {
                    "document_title": document["title"],
                    "document_chunk": " ".join(document_chunk),
                }
                chunked_nq_train_dataset.append(chunked_example)

    chunked_nq_train_dataset = Dataset.from_list(chunked_nq_train_dataset)
    chunked_nq_train_dataset.save_to_disk(os.path.join(args.save_location, "natural_questions_train_chunked_documents"))
    with open(os.path.join(args.save_location, "natural_questions_train_document_titles.json"), 'w') as outfile:
        json.dump(list(train_titles), outfile)

    validation_titles = set()
    nq_validation_dataset = load_dataset("natural_questions", split="validation", cache_dir=args.save_location)
    chunked_nq_validation_dataset = []
    for example in tqdm(nq_validation_dataset):
        document = example["document"]
        if document["title"] not in train_titles and document["title"] not in validation_titles:
            validation_titles.add(document["title"])
            tokens = document["tokens"]
            tokens_filtered = [
                token for token, is_html in zip(tokens["token"], tokens["is_html"]) if not is_html
            ]
            document_tokens = document["title"].split(" ") + tokens_filtered # prepend title

            document_chunks = split_document(document_tokens, args.chunk_size_in_tokens)
            for document_chunk in document_chunks:
                chunked_example = {
                    "document_title": document["title"],
                    "document_chunk": " ".join(document_chunk),
                }
                chunked_nq_validation_dataset.append(chunked_example)

    chunked_nq_validation_dataset = Dataset.from_list(chunked_nq_validation_dataset)
    chunked_nq_validation_dataset.save_to_disk(os.path.join(args.save_location, "natural_questions_validation_chunked_documents"))
    with open(os.path.join(args.save_location, "natural_questions_validation_document_titles.json"), 'w') as outfile:
        json.dump(list(validation_titles), outfile)
