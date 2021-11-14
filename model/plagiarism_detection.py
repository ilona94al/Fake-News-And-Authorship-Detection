from constants import PLOTS_PATH
from model.detection import Detection
import numpy as np
import matplotlib.pyplot as plt


class PlagiarismDetection(Detection):
    def __init__(self, input, model_name, author_name):
        super().__init__(input=input, model_name=model_name, model_type="Plagiarism")
        self.author_name=author_name


    def create_distribution_plot(self):
        real_percent, fake_percent = self.get_distribution()
        # Creating dataset
        authors = [self.author_name, 'Other']
        data = [real_percent, fake_percent]
        # Creating explode data
        explode = (0.2, 0.2)
        # Creating color parameters
        colors = ("darkgreen", "orange")
        # Wedge properties
        wp = {'linewidth': 1, 'edgecolor': "grey"}

        # Creating autocpt arguments
        def func(pct):
            return "{:.1f}%\n".format(pct)

        # Creating plot
        fig, ax = plt.subplots(figsize=(5, 3))
        wedges, texts, autotexts = ax.pie(data,
                                          autopct=lambda pct: func(pct),
                                          explode=explode,
                                          labels=authors,
                                          pctdistance=0.7,
                                          colors=colors,
                                          startangle=40,
                                          wedgeprops=wp,
                                          textprops=dict(color="black")
                                          )
        # Adding legend
        ax.legend(wedges, authors,
                  title="Authors",
                  loc="upper right",
                  bbox_to_anchor=(0.1, 1, 0, 0))
        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title("Distribution Graph")
        plt.savefig("../"+self.plot_path2)

    def get_distribution(self):
        real_numb = np.count_nonzero(self.predictions == 0)
        fake_numb = np.count_nonzero(self.predictions == 1)
        all = np.size(self.predictions)
        real_percent = 100 * real_numb / all
        fake_percent = 100 * fake_numb / all
        return real_percent, fake_percent

    def create_probabilities_plot(self):
        X_axis = np.arange(self.number_of_chunks)+1
        plt.bar(X_axis - 0.2, self.probabilities[:, 0], 0.4, label='Shakespeare', color="lightblue")
        plt.bar(X_axis + 0.2, self.probabilities[:, 1], 0.4, label='Others', color="salmon")
        plt.xlabel("Book chunks")
        plt.ylabel("Probability")
        plt.title("Probabilities that chunk written by " + self.author_name + " and by other writers")
        plt.legend()
        plt.savefig("../"+self.plot_path1)
