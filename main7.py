#
import matplotlib.pyplot as plt
import numpy as np

# y=np.arange(1,Y_predicted_prob.shape[0]+1)

# # plt.plot(Y_predicted_prob[0])
# # plt.plot(Y_predicted_prob[1])
# plt.bar(y,Y_predicted_prob[:,0])
# plt.bar(y,Y_predicted_prob[:,1])
# plt.title('True/Fake prob in book parts')
# plt.ylabel('Prob')
# plt.xlabel('Part')
# plt.legend(['Real', 'Fake'], loc='upper left')
# plt.show()
from constants import PLOTS_PATH
#
# Y_predicted_prob=np.random.rand(10,10)
#
# X_axis = np.arange(Y_predicted_prob.shape[0])+1
#
# plt.bar(X_axis-0.2,Y_predicted_prob[:,0], 0.4, label='Real',color="lightblue")
# plt.bar(X_axis+0.2,Y_predicted_prob[:,1], 0.4, label='Fake',color="salmon")
#
# plt.xlabel("Book parts")
# plt.ylabel("Prob")
# plt.title("---")
#
#
# plt.legend()
#
#
# plt.savefig(PLOTS_PATH+"test2.PNG")
#
#
#
# # data to display on plots
# x = ["1", "2"]
# # this will explode the 1st wedge
# # i.e. will separate the 1st wedge
# # from the chart
# e  =(0.1, 0.9)
# # This will plot a simple pie chart
# plt.pie(x, explode = e)
# # Title to the plot
# plt.title("Pie chart")
# plt.show()
# print("f")
#
# # Import libraries
# from matplotlib import pyplot as plt
# import numpy as np
#
# # Creating dataset
# authors = ['AUDI', 'BMW']
#
# data = [23, 77]
#
# # Creating plot
# fig = plt.figure(figsize=(10, 7))
# plt.pie(data, labels=authors)
#
# # show plot
# plt.show()

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Creating dataset
authors = ['Shakespeare1234556789871116', 'Other11']

data = [3, 97]

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
fig, ax = plt.subplots(figsize=(7, 5))
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
ax.legend(authors,
          title="Authorss",
          ncol=2,bbox_to_anchor =(0.5, 0.02,0.5,0))
plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Distribution Graphh")

# show plot
plt.savefig(PLOTS_PATH + "test.PNG")

plt.show()
#
#
# Y_predicted_prob=np.random.rand(10,10)
#
# X_axis = np.arange(Y_predicted_prob.shape[0])+1
#
# plt.bar(X_axis-0.2,Y_predicted_prob[:,0], 0.4, label='Real',color="lightblue")
# plt.bar(X_axis+0.2,Y_predicted_prob[:,1], 0.4, label='Fake',color="salmon")
#
# plt.xlabel("Book parts")
# plt.ylabel("Prob")
# plt.title("---")
#
#
# plt.legend()
# plt.show()
#
#
# plt.savefig(PLOTS_PATH+"test2.PNG")
