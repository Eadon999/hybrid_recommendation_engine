#!/usr/bin/env python3

import logging
import operator
import os
import sys
import time

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import sklearn.metrics as metrics
from pandas import read_csv
from scipy.sparse.linalg import spsolve
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    filename="log/mf_itemsim.log", level=logging.INFO
)


class Dataset:
    """Dataset used to calculate user recommendations"""

    def __init__(
        self,
        advertiser,
        item_user_matrix,
        item_dict,
        user_dict,
        expired_items,
        user_sale_dict=None,
    ):
        """
            Constructor.
            Params:
                advertiser = the advertiser name
                item_user_matrix = matrix with scores for item-user interactions
                item_dict = dictionary that maps external to internal item codes
                user_dict = dictionary that maps external to internal user codes
                expired_items = item IDs set to expired
                user_sale_dict = dictionary to keep track of the sale record
                    for users without duplicates
        """
        self.advertiser = advertiser
        self._item_user_matrix = item_user_matrix
        self._item_dict = item_dict
        self._user_dict = user_dict
        self._expired_items = expired_items

        # Needed for unit tests
        if user_sale_dict is None:
            user_sale_dict = {
                user: self._item_user_matrix.T[user].tocoo().col
                for user in range(self._item_user_matrix.shape[1])
            }

        self._user_sale_dict = user_sale_dict

    """Getter of members of the class."""

    """Matrix with scores for item-user interactions"""
    item_user_matrix = property(operator.attrgetter("_item_user_matrix"))
    """Dictionary that maps external to internal item codes"""
    item_dict = property(operator.attrgetter("_item_dict"))
    """dictionary that maps external to internal user codes"""
    user_dict = property(operator.attrgetter("_user_dict"))
    """Item IDs set to expired"""
    expired_items = property(operator.attrgetter("_expired_items"))
    """Dictionary to keep track of the sale record for users without duplicates
    """
    user_sale_dict = property(operator.attrgetter("_user_sale_dict"))

    def __str__(self):

        ret = "\n"
        ret += "Advertiser: %s\n" % self.advertiser
        user_count = self.item_user_matrix.shape[1]
        item_count = self.item_user_matrix.shape[0]
        ret += "Number of users: %s\n" % user_count
        ret += "Number of items: %s\n" % item_count
        num_interactions = self.item_user_matrix.count_nonzero()
        ret += "Number of interactions/ratings: %s\n" % num_interactions
        # np.float64 doesn't throw on division by zero
        ret += "Items per user: %s\n" % (
            np.float64(num_interactions) / user_count
        )
        ret += "Number of expired items: %s\n" % len(self.expired_items)
        ret += "Percentage of expired items: %s\n" % (
            np.float64(len(self.expired_items))
            / (item_count + len(self.expired_items))
            * 100
        )
        sparsity = 100 * (
            1 - (np.float64(num_interactions) / (user_count * item_count))
        )
        ret += "Sparsity: %s\n" % sparsity
        return ret


def processDataset(advertiser):
    """
        Parses the dataset (file written by nautica) and loads it into a
        list of interactions formatted as follows:
            [[users], [items], [view], [basket], [sale], [expired]]
        The function also maps external to internal user and item codes due to
        contiguous ranges of IDs are needed for matrix factorization.
        Params:
            advertiser = the advertiser name
        Returns:
            the dataset object or None if there are no users or no items to
            process.
            item-user_matrix = sparse matrix to calculate user and item embeddings through MF 
            item_dict = dictionary that maps external to internal item codes
            user_dict = dictionary that maps external to internal user codes
    """

    file_name = "data/%s_mf.dataset" % advertiser

    users = []
    items = []
    scores = []

    user_dict = {}
    item_dict = {}
    user_sale_dict = {}
    expired_items = set()

    with open(file_name, "r") as file:
        for line in file:
            values = line.strip().split(",")

            if len(values) != 6:
                continue

            user = int(values[3])
            item = int(values[4])
            view = float(values[2])
            basket = float(values[0])
            sale = float(values[1])
            expired = float(values[5])
            score = float(view + basket + sale)

            # Fix basket view issue where items are over-counted
            if basket > 1:
                basket = 1

            # Ideally the expired items should be taken into consideration
            # when training the model and filtered out to generate the user
            # recommendations. Unfortunately filtering a large number of
            # expired items is computationally expensive for advertisers with
            # a high number of items, therefore the expired items won't be
            # added to the dataset for training the model.
            # Note that the external codes are added to the expired items
            if expired > 0:
                expired_items.add(item)
                continue

            # Mapping external to internal user and item codes
            if user not in user_dict:
                user_dict[user] = len(user_dict)
            user_code = user_dict[user]  # store internal ID

            if item not in item_dict:
                item_dict[item] = len(item_dict)
            item_code = item_dict[item]  # store internal ID

            users.append(user_code)
            items.append(item_code)
            scores.append(score)

            if sale > 0:
                if user_code not in user_sale_dict:
                    user_sale_dict[user_code] = set([item_code])
                else:
                    user_sale_dict[user_code].add(item_code)

        file.close()

    if not items or not users:
        # Can't create CSR matrix if there are no users or no items to process
        return

    item_user_matrix = sparse.csr_matrix((scores, (users, items)))
    dataset = Dataset(
        advertiser,
        item_user_matrix,
        item_dict,
        user_dict,
        expired_items,
        user_sale_dict,
    )

    return dataset, item_user_matrix, item_dict, user_dict


def get_recommendation(
    item_user_matrix, item_dict, alpha, factors, top_items=30
):
    """
            Function to create 30 top recommendation for users.
            Params:
                item_user_matrix = matrix with scores for item-user interactions
                item_dict = dictionary that maps external to internal item codes
                user_dict = dictionary that maps external to internal user codes
                alpha = linear conf index to multiply
                factors = latent feature dimension of item and user embeddings
                top_items=30
            Returns:
            top 30 item recommendation for each user.    
                
    """

    user_vecs, item_vecs = implicit.alternating_least_squares(
        (item_user_matrix * alpha).astype("double"),
        factors=factors,
        regularization=0.1,
        iterations=100,
    )
    indexes = list(range(user_vecs.shape[0]))
    cos_similarity = cosine_similarity(user_vecs[indexes, :], item_vecs)
    nn = np.argsort(-cos_similarity)[:, :top_items]
    item_reco = []
    item_dict = {v: k for k, v in item_dict.items()}
    for i in range(len(nn)):
        item_reco.append([item_dict[x] for x in nn[i]])
    return item_reco


def main():
    start_time = time.time()
    advertiser = sys.argv[1]
    # inputFilePath = 'data/%s_mf.dataset' % advertiser
    userFilePath = "data/%s_user_embedding" % advertiser
    itemFilePath = "data/%s_item_embedding" % advertiser
    final_recommendation_path = "data/%s_recommendations.user" % advertiser

    print("Read files")
    # ratingMatrix is an adjacency list with each row (user, item, rating)
    dataset, item_user_matrix, item_dict, user_dict = processDataset(advertiser)
    print("Run models")
    item_reco = np.array(
        get_recommendation(item_user_matrix, item_dict, 20, 64, top_items=30)
    )

    print("Time to train model--- %s seconds ---" % (time.time() - start_time))
    print("Write embedding results")
    start_time = time.time()
    print("loading embedding weights to compute weights")
    user_dict = {v: k for k, v in user_dict.items()}
    user_id = pd.Series(list(user_dict.values())).values
    user_id = user_id[:, np.newaxis]
    item_reco = pd.concat(
        [
            pd.DataFrame(user_id.reshape(-1)).reset_index(drop=True),
            pd.DataFrame(item_reco).reset_index(drop=True),
        ],
        axis=1,
    )
    item_reco = np.array(item_reco.astype(int))
    print(
        "Time to compute recommendations--- %s seconds ---"
        % (time.time() - start_time)
    )
    np.savetxt(
        final_recommendation_path,
        np.array(item_reco),
        fmt="%d",
        header="",
        delimiter=",",
    )
    with open(final_recommendation_path + ".ready", "w") as file_ready:
        file_ready.close()
    logging.info(dataset)
    print("OK")


if __name__ == "__main__":
    main()
