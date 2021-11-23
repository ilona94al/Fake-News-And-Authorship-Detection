from model.detection import Detection
import numpy as np
import matplotlib.pyplot as plt


class FakeNewsDetection(Detection):
    def __init__(self, input, model_name):
        super().__init__(input=input, model_name=model_name, model_type="FakeNews")

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
        fig, ax = plt.subplots(figsize=(5, 3))
        wedges, texts, autotexts = ax.pie(data,
                                          autopct=lambda pct: func(pct),
                                          explode=explode,
                                          labels=categories,
                                          pctdistance=0.7,
                                          colors=colors,
                                          startangle=40,
                                          wedgeprops=wp,
                                          textprops=dict(color="black", size=10)
                                          )
        # Adding legend
        ax.legend(wedges, categories,
                  title="Categories",
                  loc="upper right",
                  bbox_to_anchor=(0.1, 1, 0, 0))
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title("Pie chart of tweet classification:", size=10)

        plt.savefig("../" + self.plot_path2)

    def create_probabilities_plot(self):

        plt.figure()
        X_axis = np.arange(self.number_of_chunks) + 1
        plt.bar(X_axis - 0.2, self.probabilities[:, 0], 0.4, label='Real', color="lightblue")
        plt.bar(X_axis + 0.2, self.probabilities[:, 1], 0.4, label='Fake', color="salmon")
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
