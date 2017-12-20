import numpy as np
import matplotlib.pyplot as plt

#train_dir = 'home/mcdonald/Dropbox/Cho/'
train_dir = ''
eval_history = np.load(train_dir+'eval_history.npy')
loss_history = np.load(train_dir+'loss_history.npy')
labels = ['teacher', 'large', 'medium', 'small', 'med labels', 'small labels',
  'med student late', 'small student late', 'med labels late', 'small labels late']
colors = ['b', 'lime', 'm', 'k', 'r', 'c', 'g', 'b', 'orange', 'hotpink']

print(eval_history.shape)
x = np.arange(eval_history.shape[1])*100.
m = 11
conv = np.ones(m)/m

plt.figure(1)
start = 900
end = 1900
for i in [0,1,2,6,7,8,9]:
#for i in range(6):
  #plt.plot(x[start:end], eval_history[i][start:end], label=labels[i], color=colors[i])
  #plt.plot(x, eval_history[i], label=labels[i])
  eval_history[i] = np.convolve(eval_history[i], conv, mode='same')
  plt.plot(x[start:end], eval_history[i][start:end], label=labels[i], color=colors[i])

plt.xlim(start*100, end*100)
plt.ylim(0, 0.9)
plt.title('Accuracy on Validation Set')
plt.xlabel('Batches (100 images/batch)')
plt.ylabel('Accuracy')
#plt.xticks(np.arange(10)*100000)
plt.legend(loc=(0.4,0.2))
plt.show()

plt.figure(2)
for i in range(4):
  #plt.plot(loss_history[i][:100000], label=labels[i])
  plt.plot(loss_history[i], label=labels[i])

plt.title('Value of Loss Function')
plt.xlabel('Batches (100 images/batch)')
plt.ylabel('Loss Function')
plt.legend(loc=0)
plt.show()


