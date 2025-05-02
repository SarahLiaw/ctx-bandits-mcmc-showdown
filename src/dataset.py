"""
Line format for yahoo events:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000
Some log files contain rows with erroneous data.
After the first 10 columns are the articles and their features.
Each article has 7 columns (articleid + 6 features)
Therefore number_of_columns-10 % 7 = 0
"""
import random
import time
from tqdm import tqdm
import torch
import numpy as np
import fileinput

class Dataset(object):
    def __init__(self):
        self.articles, self.features, self.events = [], [], [], 

    def get_yahoo_events(self, filenames, limit_size=None):
        skipped = 0
        features = []
        with fileinput.input(files=filenames) as f:
            for line in f:
                cols = line.split()
                if (len(cols) - 10) % 7 != 0:
                    skipped += 1
                else:
                    pool_idx = []
                    pool_ids = []
                    for i in range(10, len(cols) - 6, 7):
                        id = cols[i][1:]
                        if id not in self.articles:
                            self.articles.append(id)
                            features.append([float(x[2:]) for x in cols[i + 1: i + 7]])
                        pool_idx.append(self.articles.index(id))
                        pool_ids.append(id)

                    self.events.append(
                        [
                            pool_ids.index(cols[1]),
                            int(cols[2]),
                            [float(x[2:]) for x in cols[4:10]],
                            pool_idx,
                        ]
                    )
                    if limit_size != None and len(self.events) == limit_size:
                        break

        self.features = np.array(features)
        self.n_arms = len(self.articles)
        self.n_events = len(self.events)
        print(self.n_events, "events with", self.n_arms, "articles")
        if skipped != 0:
            print("Skipped events:", skipped)


    def max_articles(self, n_articles):
        assert n_articles < n_arms
        n_arms = n_articles
        articles = articles[:n_articles]
        features = features[:n_articles]

        for i in reversed(range(len(self.events))):
            displayed_pool_idx = self.events[i][0]  # index relative to the pool
            displayed_article_idx = self.events[i][3][
                displayed_pool_idx
            ]  # index relative to the articles

            if displayed_article_idx < n_arms:
                self.events[i][0] = displayed_article_idx
                self.events[i][3] = np.arange(0, n_arms)  # pool = all available articles
            else:
                del self.events[i]

        n_events = len(self.events)
        print("Number of events:", n_events)

