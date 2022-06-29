import sys
import pandas as pd
from sklearn.utils import shuffle
import random

filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = sys.argv[3]

data1 = shuffle(pd.read_csv(filename1))
data2 = shuffle(pd.read_csv(filename2))
data3 = shuffle(pd.read_csv(filename3))

user1, user2, user3 = data1["User"][0], data2["User"][0], data3["User"][0]

balance_number_of_tweets = min(data1.shape[0], data2.shape[0], data3.shape[0])

balanced_data3 = data3.sample(balance_number_of_tweets)
balanced_data1 = data1.sample(balance_number_of_tweets)
balanced_data2 = data2.sample(balance_number_of_tweets)

balanced_dataset = pd.concat([balanced_data1, balanced_data2, balanced_data3]);

dataset = pd.concat([data1, data2, data3]);

shuffled_dataset = shuffle(dataset, random_state = random.randint(1000, 100000))
shuffled_balanced_dataset = shuffle(balanced_dataset, random_state = random.randint(1000, 100000))

print (shuffled_dataset)
print (shuffled_balanced_dataset)

shuffled_dataset.to_csv("./Datasets/{}_{}_{}_dataset.csv".format(user1,user2,user3))
shuffled_balanced_dataset.to_csv("./Datasets/{}_{}_{}_balanced_dataset.csv".format(user1,user2,user3))