
import matplotlib.pyplot as plt
import numpy as np
Y_predicted_prob=np.random.rand(5,2)

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



X_axis = np.arange(Y_predicted_prob.shape[0])+1

plt.bar(X_axis-0.2,Y_predicted_prob[:,0], 0.4, label='Real')
plt.bar(X_axis+0.2,Y_predicted_prob[:,1], 0.4, label='Fake')


plt.xlabel("Book parts")
plt.ylabel("Prob")
plt.title("---")
plt.legend()
plt.show()

print("f")