# Naive_Bayes_Classifier.py

import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        words = None          # set of words

        spam_dict = None
        ham_dict = None
        Num_words_spam = 0
        Num_words_ham = 0

    def train(self, data):
        spam_dict = {}
        ham_dict = {}
        Num_words_spam = 0
        Num_words_ham = 0

        self.words = set(data.columns[1:-1])
        for word in self.words:
            spam_dict[word] = 0
            ham_dict[word] = 0

        data_spam = data[data["Prediction"] == 1]
        data_ham = data[data["Prediction"] == 0]

        for word in self.words:
            sum_spam = data_spam[word].sum()
            sum_ham = data_ham[word].sum()
            spam_dict[word] += sum_spam
            ham_dict[word] += sum_ham

            Num_words_spam += sum_spam
            Num_words_ham += sum_ham

        self.spam_dict = spam_dict
        self.ham_dict = ham_dict
        self.Num_words_spam = Num_words_spam
        self.Num_words_ham = Num_words_ham

    def predict(self, email_set):
        spam_prob = 0
        ham_prob = 0

        n = len(email_set)

        for word in email_set:
            if word not in self.words:
                continue
            spam_prob += np.log((self.spam_dict[word]+1)/(self.Num_words_spam + n))
            ham_prob += np.log((self.ham_dict[word]+1)/(self.Num_words_ham + n))

        total = self.Num_words_ham + self.Num_words_spam
        spam_prob += np.log(self.Num_words_spam/total)
        ham_prob += np.log(self.Num_words_ham/total)

        return (spam_prob, ham_prob, spam_prob > ham_prob)
    
    def Test(self, test_data):
        y_test = test_data["Prediction"]
        y_pred = []

        words = test_data.columns[1:-1]
        arr = test_data.loc[:, words].to_numpy()

        for row in arr:
            row_set = set()
            for i in range(row.shape[0]):
                if row[i] > 0:
                    row_set.add(words[i])
            
            _, _, is_spam = self.predict(row_set)
            if is_spam:
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        data = self.metric(y_test, y_pred)

        return data
    
    def metric(self, y_pred, y_test):
        true_pos = np.sum((y_pred == 1) & (y_test == 1))
        true_neg = np.sum((y_pred == 0) & (y_test == 0))
        false_pos = np.sum((y_pred == 1) & (y_test == 0))
        false_neg = np.sum((y_pred == 0) & (y_test == 1))
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1_score = 2 * precision * recall / (precision + recall)

        return accuracy, precision, recall, f1_score, true_pos, true_neg, false_pos, false_neg
