import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix, coo_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from lightfm import LightFM


class MainRecommenderALS:
    """Рекоммендации, которые можно получить из ALS
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data: pd.DataFrame, weighting: bool = True, IIR: bool  = True, ALS: bool  = False):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        if ALS:
            self.model = self.fit(self.user_item_matrix)
        
        if IIR:
            self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data: pd.DataFrame):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=256, regularization=0.001, iterations=20, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads,
                                        random_state=42,
                                        use_gpu = False
                                        )
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=False)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user_id, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user_id].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user_id, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user_id], N=N + 1)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for _user_id in similar_users:
            res.extend(self.get_own_recommendations(_user_id, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


class MainRecommenderLightFM:
    """Рекоммендации, которые можно получить из LightFM
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data_train: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame, k=50):
        
        self.user_item_matrix, self.sparse_user_item = self._prepare_matrix(data_train)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        self.user_feat_lightfm, self.item_feat_lightfm = self._features_matrix(self.user_item_matrix, user_features, item_features)

        self.model = self.fit(k)


    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def _prepare_matrix(data_train: pd.DataFrame):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data_train,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit
        
        # переведем в формат sparse matrix
        sparse_user_item = csr_matrix(user_item_matrix).tocsr()
        
        return user_item_matrix, sparse_user_item

    @staticmethod
    def _features_matrix(user_item_matrix: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame):
        """Готовит user-item-features матрицу"""
        user_feat = pd.DataFrame(user_item_matrix.index)
        user_feat = user_feat.merge(user_features, on='user_id', how='left')
        user_feat.set_index('user_id', inplace=True)

        item_feat = pd.DataFrame(user_item_matrix.columns)
        item_feat = item_feat.merge(item_features, on='item_id', how='left')
        item_feat.set_index('item_id', inplace=True)

        user_feat_lightfm = pd.get_dummies(user_feat, columns=user_feat.columns.tolist())
        item_feat_lightfm = pd.get_dummies(item_feat, columns=item_feat.columns.tolist())

        return user_feat_lightfm, item_feat_lightfm

    def fit(self, no_components=62, learning_rate=0.001, item_alph=0.3, user_alpha=0.1, k=50):
        """Обучает LightFM"""
        model = LightFM(no_components=no_components,
                        loss='warp', # "logistic","bpr"
                        learning_rate=learning_rate, 
                        item_alpha=item_alph,
                        user_alpha=user_alpha, 
                        random_state=42,
                        k=k,
                        n=k*3,
                        max_sampled=100)

        model.fit((self.sparse_user_item > 0) * 1,  # user-item matrix из 0 и 1
                    sample_weight=coo_matrix(self.user_item_matrix),
                    user_features=csr_matrix(self.user_feat_lightfm.values).tocsr(),
                    item_features=csr_matrix(self.item_feat_lightfm.values).tocsr(),
                    epochs=20, 
                    num_threads=20,
                    verbose=True)

        return model

    def _update_dict(self, user_id, item_id):
        """Если появился новыю user / item, то нужно обновить словари"""
        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

        if item_id not in self.itemid_to_id.keys():
            max_id = max(list(self.itemid_to_id.values()))
            max_id += 1
            self.itemid_to_id.update({item_id: max_id})
            self.id_to_itemid.update({max_id: item_id})

    def get_recommendations(self, data_test: pd.DataFrame):
        """Рекомендации через стардартные библиотеки implicit"""       
        
        for user, item in data_test[['user_id', 'item_id']].values:
            self._update_dict(user_id=user, item_id=item)

        # подготавливаемм id для юзеров и товаров в порядке пар user-item
        users_ids_row = data_test['user_id'].apply(lambda x: self.userid_to_id[x]).values.astype(int)
        items_ids_row = data_test['item_id'].apply(lambda x: self.itemid_to_id[x]).values.astype(int)

        # модель возвращает меру/скор похожести между соответствующим пользователем и товаром
        predictions = self.model.predict(user_ids=users_ids_row,
                                    item_ids=items_ids_row,
                                    user_features=csr_matrix(self.user_feat_lightfm.values).tocsr(),
                                    item_features=csr_matrix(self.item_feat_lightfm.values).tocsr(),
                                    num_threads=10)
        
        return predictions


class DataPreprocessing():
    def __init__(self):
        self.df_item_price = None
        self.df_user_average_check = None
        self.df_usernaverage_purchases = None
        self.df_com_price = None
        self.df_com_item = None
        self.df_commodity_desc = None

    def fit(self, df_ranker_train, df_train_matcher, item_features):
        
        df_ranker_train = df_ranker_train.copy()
        df_train_matcher = df_train_matcher.copy()
        item_features = item_features.copy()

        #заполняем переменные класса
        #добавим признаки товара
        df_ranker_train = df_ranker_train.merge(item_features, on='item_id', how='left')
        
        #цена товара
        df_train_matcher['price'] = df_train_matcher['sales_value'] / df_train_matcher['quantity']
        df_train_matcher.loc[df_train_matcher['price']==np.inf] = 0
        self.df_item_price = df_train_matcher[['user_id', 'item_id', 'price']]
        df_ranker_train = df_ranker_train.merge(self.df_item_price.groupby(['item_id'])['price'].agg('mean'), on='item_id', how='left')
        
        #Средний чек
        self.df_user_average_check = df_train_matcher.groupby(['user_id'])['price'].agg('mean').reset_index()
        self.df_user_average_check.rename(columns={'price': 'average_check'}, inplace=True)
        #df_ranker_train = df_ranker_train.merge(self.df_user_average_check, on='user_id', how='left')
        
        #Среднее количество пакупок
        self.df_usernaverage_purchases = df_train_matcher.groupby(['user_id'])['quantity'].agg('mean').reset_index()
        self.df_usernaverage_purchases.rename(columns = {'quantity': 'naverage_purchases'}, inplace=True)
        #df_ranker_train = df_ranker_train.merge(self.df_usernaverage_purchases, on='user_id', how='left')
        
        #сумма покупок в группе
        #print(df_ranker_train)
        self.df_com_price = df_ranker_train.groupby(['commodity_desc'])['price'].agg('mean').reset_index()
        self.df_com_item = df_ranker_train.groupby(['commodity_desc'])['item_id'].nunique().reset_index()
        self.df_com_item['purchases_group'] = self.df_com_item['item_id'] * self.df_com_price['price']
        df_ranker_train= df_ranker_train.merge(self.df_com_item[['commodity_desc', 'purchases_group']], on='commodity_desc', how='left')

        #Количество купленного товара в категории
        self.df_commodity_desc = df_ranker_train.groupby(['user_id', 'commodity_desc'])['item_id'].nunique().reset_index()
        self.df_commodity_desc .rename(columns={'item_id': 'commodity_item'}, inplace=True)
        #df_ranker_train = df_ranker_train.merge(self.df_commodity_desc, on=['user_id', 'commodity_desc'], how='left')

        #Количество купленного товара по брэндам
        self.df_brand_item = df_ranker_train.groupby(['user_id', 'brand'])['item_id'].nunique().reset_index()
        self.df_brand_item.rename(columns={'item_id': 'brand_item'}, inplace=True)
        #df_ranker_train = df_ranker_train.merge(self.df_brand_item, on=['user_id', 'brand'], how='left')

    def transform(self, df_ranker_train, item_features):

        df_ranker_train = df_ranker_train.copy()
        #добавим признаки товара
        df_ranker_train = df_ranker_train.merge(item_features, on='item_id', how='left')
        #добавим цену товара
        df_ranker_train = df_ranker_train.merge(self.df_item_price.groupby(['item_id'])['price'].agg('mean'), on='item_id', how='left')
        #Средний чек
        df_ranker_train = df_ranker_train.merge(self.df_user_average_check, on='user_id', how='left')
        #Среднее количество пакупок
        df_ranker_train = df_ranker_train.merge(self.df_usernaverage_purchases, on='user_id', how='left')
        #сумма покупок в группе
        df_ranker_train= df_ranker_train.merge(self.df_com_item[['commodity_desc', 'purchases_group']], on='commodity_desc', how='left')
        #Количество купленного товара в категории
        df_ranker_train = df_ranker_train.merge(self.df_commodity_desc, on=['user_id', 'commodity_desc'], how='left')
        #Количество купленного товара по брэндам
        df_ranker_train = df_ranker_train.merge(self.df_brand_item, on=['user_id', 'brand'], how='left')

        col=['department', 'brand', 'commodity_desc', 'sub_commodity_desc', 'curr_size_of_product']
        df_ranker_train[col] = df_ranker_train[col].astype('category')
        return df_ranker_train




