import numpy as np
import pandas as pd

import requests
import os   

from Naive_Bayes_Classifier import NaiveBayesClassifier

def main():
    url = "https://www.kaggle.com/api/v1/datasets/download/balaka18/email-spam-classification-dataset-csv"

    dataset_file = None
    try:
        dataset_file = Download_dataset(url)
    except:
        return 0
    
    data = pd.read_csv(dataset_file)

    TRAIN_SIZE = 0.8
    TRAIN_SPLIT = int(TRAIN_SIZE * len(data))

    data = data.sample(frac=1)        # randomization
    train_data = data[:TRAIN_SPLIT]
    test_data = data[TRAIN_SPLIT:]

    Email_spam_classifier = NaiveBayesClassifier()

    print("<--- Training --->")
    Email_spam_classifier.train(train_data)
    print("<--- Training Done --->")


    print("<--- Testing --->")
    accuracy, precision, recall, f1_score, true_pos, true_neg, false_pos, false_neg = Email_spam_classifier.Test(test_data)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)
    print("True positive: ", true_pos)
    print("True negative: ", true_neg)
    print("False positive: ", false_pos)
    print("False negative: ", false_neg)

def Download_dataset(url):
    try:
        if os.path.exists("emails.csv"):
            print("Dataset already present, Skipping download dataset.")
            return "emails.csv"
        
        print("Dataset not present, downloading dataset...")
        response = requests.get(url)
        with open("dataset.zip", "wb") as f:
            f.write(response.content)

        os.system("unzip dataset.zip emails.csv")
        os.remove("dataset.zip")

        print("Download Done!!")
        return "emails.csv"
    except:
        print("Download failed!!")
        raise Exception("Download failed!!")


if __name__ == "__main__":
    main()