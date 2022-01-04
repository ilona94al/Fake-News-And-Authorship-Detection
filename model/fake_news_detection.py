from constants import RESULTS
from model.detection import Detection
import numpy as np
import matplotlib.pyplot as plt


class FakeNewsDetection(Detection):
    def __init__(self, input, model_name):
        super().__init__(input=input, model_name=model_name, model_type="FakeNews")

        results_csv_path =RESULTS+ str(model_name).removesuffix('.h5')+'- Detection results.csv'
        self.write_results_to_file(file_path=results_csv_path, text=input, real_percent=self.real_percent)

    def create_distribution_plot(self):
        import matplotlib.pyplot as plt
        # Creating dataset
        categories = ['Real', 'Fake']
        data = [self.real_percent, self.fake_percent]
        # Creating explode data
        explode = (0.2, 0.2)
        # Creating color parameters
        colors = ("limegreen", "lightcoral")
        # Wedge properties
        wp = {'linewidth': 1, 'edgecolor': "black"}

        # Creating autocpt arguments
        def func(pct):
            return "{:.1f}%\n".format(pct)

        # Creating plot
        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(data,
                                          autopct=lambda pct: func(pct),
                                          explode=explode,
                                          labels=categories,
                                          pctdistance=0.7,
                                          colors=colors,
                                          startangle=40,
                                          wedgeprops=wp,
                                          textprops=dict(color="black", size=12)
                                          )
        # Adding legend
        plt.legend(wedges, categories,
                  title="Categories",
                  ncol=2,bbox_to_anchor =(0.5, 0.02,0.5,0))
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title("Pie chart of tweet classification:", size=12,weight="bold")

        plt.savefig("../" + self.plot_path2)

    def create_probabilities_plot(self):

        plt.figure()

        X_axis = np.arange(self.number_of_chunks)
        plt.bar(X_axis + 0.25, self.probabilities[:, 0], 0.5, label='Real', color="lightblue")
        plt.bar(X_axis + 0.75, self.probabilities[:, 1], 0.5, label='Fake', color="salmon")

        plt.xlabel("Tweet chunks", weight="bold")
        plt.ylabel("Probability", weight="bold")
        plt.title("Probabilities for each chunk to be real / fake ", weight="bold")
        plt.legend()

        plt.savefig("../" + self.plot_path1)

    def get_result(self):
        text = "The tweet represents fake news with an accuracy of " + "{:.1f}%".format(self.real_percent) + ". " + "\n"
        if self.real_percent > self.fake_percent:
            text += "It seems like the tweet is reliable and does not represent fake news."
        else:
            text += "It seems like the tweet isn't reliable and represents fake news."
        return text

    @staticmethod
    def write_results_to_file(file_path, text, real_percent):
        import csv
        from pathlib import Path

        my_file = Path(file_path)

        file_exist = my_file.exists()

        with open(file_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if not file_exist:
                header = ['text', 'reliable news percent', 'real classification (if exist)']
                writer.writerow(header)
            data = [text, "{:.1f}%".format(real_percent)]
            writer.writerow(data)
