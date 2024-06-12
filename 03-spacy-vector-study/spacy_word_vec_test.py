import spacy
import numpy as np

from prettytable import PrettyTable

def cosine_sim(vector_1: np.ndarray, vector_2: np.ndarray) -> np.float32:
    return np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))

if __name__ == "__main__":
    spacy_nlp = spacy.load("en_core_web_sm")
    banana_vec = spacy_nlp("banana").vector
    banAna_vec = spacy_nlp("banAna").vector
    apple_vec = spacy_nlp("apple").vector
    torpedo_vec = spacy_nlp("torpedo").vector
    

    table = PrettyTable()
    table.field_names = ["Word 1", "Word 2", "Cosine Similarity"]
    table.add_row(["banana", "banana", cosine_sim(banana_vec, banana_vec)])
    table.add_row(["banana", "banAna", cosine_sim(banana_vec, banAna_vec)])
    table.add_row(["banana", "apple", cosine_sim(banana_vec, apple_vec)])
    table.add_row(["banana", "torpedo", cosine_sim(banana_vec, torpedo_vec)])
    table.add_row(["apple", "torpedo", cosine_sim(apple_vec, torpedo_vec)])

    print(table)




