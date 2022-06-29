
from cmath import inf
import sys
import snscrape.modules.twitter as sntwitter
import pandas as pd

user = sys.argv[1]

query = "(from:" + user + ") lang:en exclude:nativeretweets exclude:retweets" # -filter:replies"

tweets = []
num_of_tweets = inf

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == num_of_tweets:
        break
    else:
        tweets.append([tweet.user.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['User', 'Tweet'])
print(df.shape)
df.to_csv("./csv_files/{}_tweets_with_replies.csv".format(user))

