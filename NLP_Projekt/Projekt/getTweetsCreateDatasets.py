from cmath import inf
import sys
import snscrape.modules.twitter as sntwitter
import pandas as pd
from sklearn.utils import shuffle
import random

def createDataset(df1, df2, df3):
    dataset = pd.concat([df1, df2, df3]);
    user1, user2, user3 = df1["User"][0], df2["User"][0], df3["User"][0]
    
    shuffled_dataset = shuffle(dataset, random_state = random.randint(1000, 100000))
    return shuffled_dataset

def createBalancedDataset(df1, df2, df3):
    user1, user2, user3 = df1["User"][0], df2["User"][0], df3["User"][0]
    
    balance_number_of_tweets = min(df1.shape[0], df2.shape[0], df3.shape[0])
    
    balanced_data1 = df1.sample(balance_number_of_tweets)
    balanced_data2 = df2.sample(balance_number_of_tweets)
    balanced_data3 = df3.sample(balance_number_of_tweets)
    
    balanced_dataset = pd.concat([balanced_data1, balanced_data2, balanced_data3]);
    
    shuffled_balanced_dataset = shuffle(balanced_dataset, random_state = random.randint(1000, 100000))
    return shuffled_balanced_dataset

def getTweets(user, retweets = False, replies = True, number_of_tweets = inf):
    query = "(from:" + user + ") lang:en " 
    if not retweets: 
        query += "exclude:nativeretweets exclude:retweets "
    if not replies: 
        query += "-filter:replies "
    
    tweets = []
    num_of_tweets = number_of_tweets

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == num_of_tweets:
            break
        else:
            tweets.append([tweet.user.username, tweet.content])
            
    columns = ['User', 'Tweet']
            
    df = pd.DataFrame(tweets, columns=columns)
    return df
    
if __name__ == '__main__':
    user1 = sys.argv[1]
    user2 = sys.argv[2]
    user3 = sys.argv[3]
    path_to_save = sys.argv[4]
    
    df1 = getTweets(user1)
    df2 = getTweets(user2)
    df3 = getTweets(user3)
    
    dataset = createDataset(df1, df2, df3)
    
    balanced_dataset = createBalancedDataset(df1, df2, df3)
    
    balanced_dataset.to_csv("./"+ path_to_save + "/{}_{}_{}_dataset.csv".format(user1,user2,user3))
    dataset.to_csv("./"+ path_to_save + "/{}_{}_{}_balanced_dataset.csv".format(user1,user2,user3))