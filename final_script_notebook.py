def get_recommendation(item_user_matrix,item_dict,alpha,factors,top_items=30):
    user_vecs, item_vecs = implicit.alternating_least_squares(
    (item_user_matrix*alpha).astype('double'), 
     factors=factors, 
     regularization = 0.1, 
     iterations = 100)
    indexes = list(range(user_vecs.shape[0]))
    cos_similarity = cosine_similarity(
        user_vecs[indexes, :], item_vecs)
    nn = np.argsort(-cos_similarity)[:, :top_items]
    item_reco = []
    item_dict = {v: k for k, v in item_dict.items()}
    for i in range(len(nn)):
        item_reco.append([item_dict[x] for x in nn[i]]) 
    return item_reco


In [88]:
def processDataset (file_name):
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
    """

    #file_name = 'data/%s_mf.dataset' % advertiser

    users = []
    items = []
    scores = []

    user_dict = {}
    item_dict = {}
    user_sale_dict = {}
    expired_items = set()

    with open(file_name, 'r') as file:
        for line in file:
            values = line.strip().split(',')

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
            user_code = user_dict[user] # store internal ID

            if item not in item_dict:
                item_dict[item] = len(item_dict)
            item_code = item_dict[item] # store internal ID

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

    item_user_matrix = sparse.csr_matrix((scores, (users,items)))

    return item_user_matrix,item_dict,user_dict

def main():
    #alpha_val = 20
    start_time = time.time()
    ratingFilePath = '/home/sourish/work/MF/outputs/';
    inputFilePath = '/home/sourish/work/MF/data/riu-de.dataset'
    userFilePath = ratingFilePath + "user_embedding";
    itemFilePath = ratingFilePath + "item_embedding";
    final_recommendation_path = ratingFilePath + "final_reccomendation_mf_item"

    print("Read files");
    # ratingMatrix is an adjacency list with each row (user, item, rating)
    item_user_matrix,item_dict,user_dict = processDataset(inputFilePath)
    #conf_data = (item_user_matrix * alpha_val).astype('double')
    print("Run models");
    item_reco = np.array(get_recommendation(item_user_matrix,item_dict,20,64,\
                                            top_items=30))
    
    print("Time to train model--- %s seconds ---" % (time.time() - start_time))
    print("Write embedding results");
    start_time = time.time()
    print('loading embedding weights to compute weights')
    user_dict = {v: k for k, v in user_dict.items()}
    user_id = pd.Series(list(user_dict.values())).values
    user_id = user_id[:,np.newaxis]
    item_reco = pd.concat([pd.DataFrame(user_id.reshape(-1)).reset_index\
    (drop=True),pd.DataFrame(item_reco).reset_index(drop=True)],axis=1)
    item_reco = np.array(item_reco.astype(int))
    print("Time to compute recommendations--- %s seconds ---" % (time.time() - start_time))
    np.savetxt(final_recommendation_path,np.array(item_reco),fmt = "%d", header = "", delimiter = ",")
    #with open(final_recommendation_path + '.ready', 'w') as file_ready:
        #file_ready.close()
    #logging.info(dataset)
    print("OK");

if __name__ == "__main__":
    main()

def get_recommendation(item_user_matrix,item_dict,alpha,factors,top_items=30):
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
    cos_similarity = cosine_similarity(
        users_vec[indexes, :], items_vec)
    nn = np.argsort(-cos_similarity)[:, :top_items]
    item_reco = []
    item_dict = {v: k for k, v in item_dict.items()}
    for i in range(len(nn)):
        item_reco.append([item_dict[x] for x in nn[i]]) 
    return item_reco
