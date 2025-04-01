import pandas as pd

from model_card import ModelCard


class ModelReport:
    def __init__(self, train, test):
        # [[TP, FN],[FP, TN]]
        self.table = pd.crosstab(train,test)
        self.table1 = [[self.table[0,0], self.table[0,1] + self.table[0,2]],
                       [self.table[1,0] + self.table[2,0], self.table[1,1] + self.table[1,2] + self.table[2,1] + self.table[2,2]]]
        self.table2 = [[self.table[1,1], self.table[1,0] + self.table[1,2]],
                       [self.table[0,1] + self.table[2,1], self.table[0,0] + self.table[2,2] + self.table[0,2] + self.table[2,0]]]
        self.table3 = [[self.table[2,2], self.table[2,1] + self.table[2,0]],
                       [self.table[1,2] + self.table[0,2], self.table[1,1] + self.table[1,0] + self.table[0,1] + self.table[0,0]]]
    def accuracy(self):
        """
        Calculate accuracy from confusion matrix (crosstab).
        Accuracy = (True predictions) / (Total predictions)
        """
        # Sum of the diagonal elements (correct predictions)
        correct_predictions = sum(self.table.values[i, i] for i in range(len(self.table)))

        # Sum of all elements (total predictions)
        total_predictions = self.table.values.sum()

        # Calculate accuracy
        return correct_predictions / total_predictions

    def precision(self,table):
        """
        Calculate precision from confusion matrix (crosstab).
        Precision = TP / (TP + FP)
        """
        tp = table[0, 0]
        fp = table[0, 0] + table[1, 0]
        return tp / (tp + fp) if (tp + fp) != 0 else 0

    def recall(self,table):
        """
        Calculate recall from confusion matrix (crosstab).
        Recall = TP / (TP + FN)
        """
        tp = table[0, 0]
        fn = table[0, 0] + table[0, 1]
        return tp / (tp + fn) if (tp + fn) != 0 else 0

    def f1_score(self,table):
        """
        Calculate F1 score from confusion matrix (crosstab).
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        """
        precision = self.precision(table)
        recall = self.recall(table)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def micro_average(self, fn):
        """
        Calculate microaveraged metric
        Args:
            fn: function to calculate Ex: (precision, recall, f1)
        Returns: microaveraged value
        """
        interim_table = self.table1 + self.table2 + self.table3
        return fn(interim_table)

    def macro_average(self, fn):
        """
        Calculate macroaveraged metric
        Args:
            fn: function to calculate Ex: (precision, recall, f1)
        Returns: macroaveraged value
        """
        return (fn(self.table1) + fn(self.table2) + fn(self.table3))/3




    def display_report(self):

        print("="*50)
        print("Model Metrics Report")
        print("="*50)

        print(f"Accuracy: {self.accuracy():.2f}")
        print(f"Macro Accuracy: {self.macro_average(self.accuracy):.2f}")

        print("="*50)

        # ⚠️ Most likely meaningless
        print(f"Micro Accuracy: {self.micro_average(self.accuracy):.2f}")

        print(f"Macro Precision: {self.macro_average(self.precision):.2f}")
        print(f"Micro Precision: {self.micro_average(self.precision):.2f}")

        print("="*50)

        print(f"Macro Recall: {self.macro_average(self.recall):.2f}")
        print(f"Micro Recall: {self.micro_average(self.recall):.2f}")

        print("="*50)

        print(f"Macro F1: {self.macro_average(self.f1_score):.2f}")
        print(f"Micro F1: {self.micro_average(self.f1_score):.2f}")


def evaluate(model : ModelCard, cache=True):
    if cache and model.report:
        print("⚠️ Using cached report ⚠️")
        return
    # evaluate model

def evaluate_models(models):
    # get train data
    # get test data
    for model in models:
        evaluate(model)
        model.display_card()
        print("="*50)

# plot models against baseline

def main():
    model_list = [
        ModelCard("Gemma", "2", "Google's largest latest open source model.",
                  "Google", 0, 27, "results/Gemma2_27b/output.csv"),
        ModelCard("Llama", "3.3 Instruct Turbo", "Meta's latest open source model.",
                  "Meta", 0, 70, "results/meta-llama_Llama-3.3-70B-Instruct-Turbo-Free/output.csv"),
    ]
    evaluate_models(model_list)


if __name__ == "__main__":
    main()
