import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_card import ModelCard


class ModelReport:
    def __init__(self, train, test):
        # [[TP, FN],[FP, TN]]
        self.table = pd.crosstab(train,test)
        self.table1 = [[self.table.iat[0,0], self.table.iat[0,1] + self.table.iat[0,2]],
                       [self.table.iat[1,0] + self.table.iat[2,0], self.table.iat[1,1] + self.table.iat[1,2] + self.table.iat[2,1] + self.table.iat[2,2]]]
        self.table2 = [[self.table.iat[1,1], self.table.iat[1,0] + self.table.iat[1,2]],
                       [self.table.iat[0,1] + self.table.iat[2,1], self.table.iat[0,0] + self.table.iat[2,2] + self.table.iat[0,2] + self.table.iat[2,0]]]
        self.table3 = [[self.table.iat[2,2], self.table.iat[2,1] + self.table.iat[2,0]],
                       [self.table.iat[1,2] + self.table.iat[0,2], self.table.iat[1,1] + self.table.iat[1,0] + self.table.iat[0,1] + self.table.iat[0,0]]]
    def accuracy(self):
        """
        Calculate accuracy from confusion matrix (crosstab).
        Accuracy = (True predictions) / (Total predictions)
        """
        # Sum of the diagonal elements (correct predictions)
        correct_predictions = sum(self.table.values[i, i] for i in range(0,2))

        # Sum of all elements (total predictions)
        total_predictions = self.table.values.sum()

        # Calculate accuracy
        return correct_predictions / total_predictions

    def precision(self,table):
        """
        Calculate precision from confusion matrix (crosstab).
        Precision = TP / (TP + FP)
        """
        tp = table[0][0]
        fp = table[0][0] + table[1][0]
        return tp / (tp + fp) if (tp + fp) != 0 else 0

    def recall(self,table):
        """
        Calculate recall from confusion matrix (crosstab).
        Recall = TP / (TP + FN)
        """
        tp = table[0][0]
        fn = table[0][0] + table[0][1]
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

        #print(f"Macro Accuracy: {self.macro_average(self.accuracy):.2f}")
        # ⚠️ Most likely meaningless
        #print(f"Micro Accuracy: {self.micro_average(self.accuracy):.2f}")
        print("="*50)


        print(f"Macro F1: {self.macro_average(self.f1_score):.2f}")
        print(f"Micro F1: {self.micro_average(self.f1_score):.2f}")

        print("="*50)

        print(f"Macro Recall: {self.macro_average(self.recall):.2f}")
        print(f"Micro Recall: {self.micro_average(self.recall):.2f}")

        print("="*50)
        print(f"Macro Precision: {self.macro_average(self.precision):.2f}")
        print(f"Micro Precision: {self.micro_average(self.precision):.2f}")

def evaluate(model : ModelCard,test,cache=True):
    if cache and model.report:
        print("⚠️ Using cached report ⚠️")
        return
    fit = pd.read_csv(model.file_path)["model_classification"]
    model.report = ModelReport(test,fit)

def plot_reports(models, plot):
    if not plot:
        return

    model_names = []
    macro_f1 = []
    model_sizes = []
    authors = []
    baselines = []

    if plot == "full":
        accuracies = []

    for model in models:
        if model.size == 0:
            baselines.append(model)
        else:
            model_names.append(model.model_name + " " + model.version)
            macro_f1.append(model.report.macro_average(model.report.f1_score))
            model_sizes.append(model.size)
            authors.append(model.author)
            if plot == "full":
                accuracies.append(model.report.accuracy())

    df = pd.DataFrame({
        'Model Name': model_names,
        'Size': model_sizes,
        'Macro F1 Score': macro_f1,
        'Author': authors
    })

    plt.figure(figsize=(12, 8))

    # Create the scatterplot
    ax = sns.scatterplot(
        data=df,
        x='Size',
        y='Macro F1 Score',
        hue='Author',
        alpha=0.9,
        palette='viridis'
    )

    for _, row in df.iterrows():
        plt.text(
            row['Size'] * 1.05,
            row['Macro F1 Score'],
            row['Model Name'],
            fontsize=9,
            alpha=0.8
        )

    #Add Base Lines
    for model in baselines:
        # Add a horizontal line for the baseline model
        plt.axhline(
            y=model.report.macro_average(model.report.f1_score),
            color='red',
            linestyle='--',
            label=f"{model.model_name} {model.version} (Baseline)"
        )

    # Set the title and labels
    plt.title('Large Language Model Performance: Macro F1 Score vs Model Size by Author', fontsize=16, pad=20)
    plt.xlabel('Model Size (Billions of Parameters)', fontsize=14)
    plt.ylabel('Macro F1 Score', fontsize=14)

    # use the range of F1 scores to set appropriate y-axis limits
    y_min = max(0, min(macro_f1) - 0.05)
    y_max = min(1.0, max(macro_f1) + 0.05)
    plt.ylim(y_min, y_max)

    # if we have a wide range of model sizes
    size_range = max(model_sizes) / max(min(model_sizes),1)
    if size_range > 10:
        # Use logarithmic scale for x-axis if sizes vary widely
        plt.xscale('log')
        plt.xlim(min(model_sizes) * 0.8, max(model_sizes) * 1.2)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations to explain the plot
    plt.figtext(
        0.5, 0.01,
        "Figure 1: Relationship between model size and macro F1 score across different language models.\n" +
        ("Note: Model size is displayed on a logarithmic scale due to the wide range of values." if size_range > 10 else ""),
        ha='center',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5)
    )

    # Improve the legend
    plt.legend(
        title='Model Developer',
        loc='best',
        frameon=True,
        framealpha=0.9
    )

    # Adjust layout to make room for annotations
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    # Show the plot
    plt.show()

    sns.barplot(df, x='Model Name', y='Macro F1 Score', hue='Size', palette='viridis')
    plt.xlabel('Model Name', fontsize=14)
    plt.ylabel('Macro F1 Score', fontsize=14)

    #Add Base Lines
    for model in baselines:
        # Add a horizontal line for the baseline model
        plt.axhline(
            y=model.report.macro_average(model.report.f1_score),
            color='red',
            linestyle='--',
            label=f"{model.model_name} {model.version} (Baseline)"
        )
    plt.title('Macro F1 Score by Model Name', fontsize=16)
    plt.show()


def evaluate_models(models, plot = False, file_path="./data/train.jsonl"):
    # get train data
    # get test data
    test = pd.read_json(path_or_buf=file_path, lines=True)["label"].reset_index(drop=True)
    for model in models:
        evaluate(model,test)
        model.display_card()
        print("="*50)

    # plot reports
    plot_reports(models, plot)

def main():
    plt.style.use("dark_background")
    model_list = [
        ModelCard("Gemma", "2", "Google's largest latest open source model.",
                  "Google", 0, 27, "results/Gemma2_27b/output.csv"),
        ModelCard("Llama", "3.3 Instruct Turbo", "Meta's latest open source model.",
                  "Meta", 0, 70, "results/meta-llama_Llama-3.3-70B-Instruct-Turbo-Free/output.csv"),
        ModelCard("Random", "Indiscriminate", "Random model.", "Nikhil",
                  0, 0,"results/Completely_random/output.csv"),
        ModelCard("Random", "Proportional", "Random model.", "Nikhil",
                  0, 0,"results/Proportionally_random/output.csv"),
        ModelCard("Single Class", "Majority", "Why even try.", "Nikhil",
                  0, 0,"results/Majority/output.csv"),
        ModelCard("Gemma", "2 Cleaned", "Google's largest latest open source model.",
                  "Google", 0, 27, "results/Gemma2_27b_clean/output.csv")
        # ModelCard("Gemma","2 No Section Name","experimenting with no section",
        #           "Google",0,27,"results/Gemma2_27b_nosectionname/fourth_partition.csv")
    ]
    evaluate_models(model_list,"brief")


if __name__ == "__main__":
    main()
