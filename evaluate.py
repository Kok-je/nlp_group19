import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from model_card import ModelCard


class ModelReport:
    def __init__(self, train, test,name,version):
        # [[TP, FN],[FP, TN]]
        self.table = pd.crosstab(train,test)
        if self.table.shape[1] != 3:
            warnings.warn(f"Invalid labels predicted for {name} {version}, might break calculations.")
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
        correct_predictions = sum(self.table.values[i, i] for i in range(0,3))

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
        fp = table[1][0]
        return tp / (tp + fp) if (tp + fp) != 0 else 0

    def recall(self,table):
        """
        Calculate recall from confusion matrix (crosstab).
        Recall = TP / (TP + FN)
        """
        tp = table[0][0]
        fn = table[0][1]
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
    assert len(fit) == len(test)
    model.report = ModelReport(test,fit,model.model_name,model.version)

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

    if plot == "full":


        plt.figure(figsize=(12, 8))

        df['Accuracy'] = accuracies

        ax = sns.scatterplot(
                data=df,
                x='Size',
                y='Accuracy',
                hue='Author',
                alpha=0.9,
                palette='viridis'
            )

        for _, row in df.iterrows():
            plt.text(
                row['Size'] * 1.05,
                row['Accuracy'],
                row['Model Name'],
                fontsize=9,
                alpha=0.8
            )

        #Add Base Lines
        for model in baselines:
            # Add a horizontal line for the baseline model
            plt.axhline(
                y=model.report.accuracy(),
                color='red',
                linestyle='--',
                label=f"{model.model_name} {model.version} (Baseline)"
            )

        # Set the title and labels
        plt.title('Large Language Model Performance: Accuracy vs Model Size by Author', fontsize=16, pad=20)
        plt.xlabel('Model Size (Billions of Parameters)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        # use the range of F1 scores to set appropriate y-axis limits
        y_min = max(0, min(accuracies) - 0.05)
        y_max = min(1.0, max(accuracies) + 0.05)
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
            "Figure 1: Relationship between model size and Accuracies across different language models.\n" +
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

def evaluate_models(models, plot = False, file_path="./data/train_cleaned.jsonl"):
    # get train data
    # get test data
    test = pd.read_json(path_or_buf=file_path, lines=True)["label"]
    if file_path == "/data/train_cleaned.jsonl":
        assert len(test) == 8194 # "Test data should be 8194 rows long"
    for model in models:
        if model.partition:
            if model.partition == 200:
                evaluate(model,test[:200])
            elif 0 < model.partition < 6:
                evaluate(model,test[(model.partition-1) * 1365 : model.partition * 1365].reset_index(drop=True))
            elif model.partition == 6:
                evaluate(model,test[5 * 1365:].reset_index(drop=True))
            else :
                print(f"Invalid partition size, skipping evaluation for {model.model_name}.")
                continue
        else:
            evaluate(model,test)
        model.display_card()
        print("="*50)

    # plot reports
    plot_reports(models, plot)

def main():
    plt.style.use("dark_background")
    model_list = [
        ModelCard("Gemma", "2", "Google's largest latest open source model.",
                  "Google", 0, 27, "results/Teachers/Gemma/Gemma2_27b/output.csv"),
        ModelCard("Llama", "3.3 Instruct Turbo", "Meta's latest open source model.",
                  "Meta", 0, 70, "results/Teachers/Llama/meta-llama_Llama-3.3-70B-Instruct-Turbo-Free/output.csv"),
        ModelCard("Random", "Indiscriminate", "Random model.", "Nikhil",
                  0, 0, "results/baselines/Completely_random/output.csv"),
        ModelCard("Random", "Proportional", "Random model.", "Nikhil",
                  0, 0, "results/baselines/Proportionally_random/output.csv"),
        ModelCard("Single Class", "Majority", "Why even try.", "Nikhil",
                  0, 0, "results/baselines/Majority/output.csv"),
        ModelCard("Gemma", "2 Cleaned", "Google's largest latest open source model.",
                  "Google", 0, 27, "results/Teachers/Gemma/Gemma2_27b_clean/output.csv"),
        ModelCard("Gemma","2 No Section Name","experimenting with no section",
                   "Google", 0, 27, "results/Teachers/Gemma/Gemma2_27b_nosectionname/fourth_partition.csv", partition = 4),
        # ModelCard("Llama", "3.3 Instruct Turbo Clean", "Meta's latest open source model.",
        #           "Meta", 0, 70, "results/Teachers/Llama/meta-llama-corrected/sixth_partition_llama.csv", partition = 6),
        # ModelCard("Llama", "3.3 Instruct Turbo No Section Name", "Meta's latest open source model.",
        #           "Meta", 0, 70,
        #           "results/Teachers/Llama/meta-llama-corrected/sixth_partition_llama_no_section_name.csv", partition = 6),
        # ModelCard("GPT", "4o Cheater", "OpenAI's latest model.", # Duplicated data
        #           "OpenAI", 0, 1000, "results/Teachers/GPT4o/cheated/output.csv"),
        # ModelCard("GPT", "4o", "OpenAI's latest model.",         # Duplicated data
        #           "OpenAI", 0, 1000, "results/Teachers/GPT4o/fair/output.csv"),
        # ModelCard("GPT", "4o api", "OpenAI's latest model.",
        #           "OpenAI", 0, 1000, "results/Teachers/GPT4o/api/output.csv",200),
        ModelCard("GPT", "4o full", "OpenAI's latest model.",
                  "OpenAI", 0, 1000, "results/Teachers/GPT4o/api/4o_full.csv"),
        # ModelCard("Gemini","2.5 Pro", "Google's latest model.",
        #           "Google", 0, 1000, "results/Teachers/Gemini-2.5-Pro/Gemini_2.5_Pro.csv"),
        ModelCard("Mistral","v1", "Mistral's first model.",
                  "Mistral", 0, 7, "results/student_models/mistral7b/first_partition_student_mistral7b.csv",1),
        ModelCard("Gemma","3", "Google's latest model.",
                  "Google", 0, 1, "results/student_models/Gemma/Gemma-1b/train/first200.csv", 200),
        ModelCard("Gemma","3 no reason", "Google's latest model.",
                  "Google", 0, 1, "results/student_models/Gemma/Gemma-1b/train/first200_no_reason.csv", 200),
        ModelCard("Gemma","3 no reason clean", "Google's latest model.",
                  "Google", 0, 1, "results/student_models/Gemma/Gemma-1b/train/first200_no_reason_clean.csv", 200),
        # ModelCard("DeepSeek","R1", "DeepSeek's latest model.",
        #           "DeepSeek", 0, 671, "results/Teachers/DeepSeek/R1/old/output_fixed.csv", 200),
        ModelCard("DeepSeek","R1 full", "DeepSeek's latest model.",
                  "DeepSeek", 0, 671, "results/Teachers/DeepSeek/R1/output.csv"),
        ModelCard("Llama","3.2", "Meta's smallest model.",
                  "Meta", 0, 1, "results/student_models/llama3.2_1b/Llama-3.2-1B-Instruct_first_partition.csv",1),
        ModelCard("Llama","3.2 clean", "Meta's smallest model.",
                  "Meta", 0, 1, "results/student_models/llama3.2_1b/Llama-3.2-1B-Instruct_first_partition_clean.csv",1),
        ModelCard("Our Teacher","v0","Our best teacher model so far",
                  "NLP Team 19", 0, 900, "results/Teachers/Ours/deepseek-openai/deepseek_openai_combined.csv"),
        ModelCard("The King","v1","is back",
                  "NLP Team 19", 0,1000,"results/Teachers/Ours/LongLiveLLama.csv")
    ]
    evaluate_models(model_list,"full")
    testing = [
    ModelCard("Gemma","3", "Google's latest model.",
              "Google", 0, 1, "results/student_models/Gemma/Gemma-1b/test/output.csv"),
    ModelCard("Gemma","3 clean", "Google's latest model.",
              "Google", 0, 1, "results/student_models/Gemma/Gemma-1b/test/output_clean.csv"),
    ModelCard("T5","non og- The King","T5 trained on non og code with The King",
              "NLP Team 19", 0, 1,"results/Trained/predictions_t5_trained.csv")]
    evaluate_models(testing, "full", "data/test.jsonl")



if __name__ == "__main__":
    main()
