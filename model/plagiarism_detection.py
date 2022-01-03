from constants import PLOTS_PATH
from model.detection import Detection
import numpy as np
import matplotlib.pyplot as plt


class PlagiarismDetection(Detection):
    def __init__(self, input, model_name, author_name, book_name):
        super().__init__(input=input, model_name=model_name, model_type="Plagiarism")

        self.author_name = author_name
        self.book_name = book_name

        results_csv_path = author_name+'- Detection results.csv'
        self.write_results_to_file(file_path=results_csv_path, author_name=author_name, book_name=book_name,
                                   percent=self.real_percent)

    def create_distribution_plot(self):
        # Creating dataset
        authors = [self.author_name, 'Other']
        data = [self.real_percent, self.fake_percent]
        # Creating explode data
        explode = (0.2, 0.2)
        # Creating color parameters
        colors = ("limegreen", "lightcoral")
        # Wedge properties
        wp = {'linewidth': 0.5, 'edgecolor': "grey"}

        # Creating autocpt arguments
        def func(pct):
            return "{:.1f}%\n".format(pct)

        # Creating plot
        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(data,
                                          autopct=lambda pct: func(pct),
                                          explode=explode,
                                          labels=authors,
                                          pctdistance=0.7,
                                          colors=colors,
                                          startangle=40,
                                          wedgeprops=wp,
                                          textprops=dict(color="black", size=12)
                                          )
        # Adding legend
        ax.legend(wedges, authors,
                  title="Authors",
                  ncol=2,bbox_to_anchor =(0.5, 0.02,0.5,0))
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title("Pie chart of book author classification:", size=12, weight="bold")
        plt.savefig("../" + self.plot_path2)

    # def get_distribution(self):
    #     real_numb = np.count_nonzero(self.predictions == 0)
    #     fake_numb = np.count_nonzero(self.predictions == 1)
    #     all = np.size(self.predictions)
    #     real_percent = 100 * real_numb / all
    #     fake_percent = 100 * fake_numb / all
    #     return real_percent, fake_percent

    def create_probabilities_plot(self):
        plt.figure()
        X_axis = np.arange(self.number_of_chunks)
        plt.bar(X_axis + 0.25, self.probabilities[:, 0], 0.5, label=self.author_name, color="lightblue")
        plt.bar(X_axis + 0.75, self.probabilities[:, 1], 0.5, label='Others', color="salmon")
        plt.xlabel("Book chunks", weight="bold")
        plt.ylabel("Probability", weight="bold")
        plt.title("Probabilities chart for book chunks writer",
                  weight="bold")
        plt.legend()
        plt.savefig("../" + self.plot_path1)

    def get_result(self):
        text = "The book: \"" + str(self.book_name) + \
               "\" was written by " + str(self.author_name) \
               + " with " + "{:.1f}%".format(self.real_percent) + " certainty." + "\n"
        if self.real_percent > self.fake_percent:
            text += "It seems like the book was written by " + self.author_name + "."
        else:
            text += "It seems like the book wasn't written by Shakespeare"
        return text

    @staticmethod
    def write_results_to_file(file_path, author_name, book_name, percent):
        import csv
        from pathlib import Path

        my_file = Path("../"+file_path)

        file_exist = my_file.exists()

        with open("../"+file_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if not file_exist:
                header = ['author name', 'book name', 'percent that written by author']
                writer.writerow(header)
            data = [author_name, book_name, "{:.1f}%".format(percent)]
            writer.writerow(data)

