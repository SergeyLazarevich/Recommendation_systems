"""
utils

"""
import pandas as pd
import numpy as np

def prefilter_items(data, 
                    take_n_popular=5000,
                    resent_weeks = 48,
                    filter_frequency = True, 
                    filter_prices = True,
                    popularity_invert = True,
                    item_features=None, 
                    user_features = None):

    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'household_key': 'user_id',
                    'product_id': 'item_id'}, inplace=True)
    if user_features is not None:
        user_features.columns = [col.lower() for col in user_features.columns]
        user_features.rename(columns={'household_key': 'user_id'}, inplace=True)

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        item_features.columns = [col.lower() for col in item_features.columns]
        item_features.rename(columns={'product_id': 'item_id'}, inplace=True)

        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size.loc[department_size['n_items'] > \
                                            np.nanquantile(department_size['n_items'], q=0.30)].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data.loc[data['item_id'].isin(items_in_rare_departments)]

    # Уберем товары с нулевым количеством и выручкой
    data = data[data['quantity'] != 0]
    data = data[data['sales_value'] != 0]

    # Рассчитаем цену товара.
    prices = data.groupby('item_id').agg({
        'sales_value' : 'sum', 
        'quantity': 'sum'
    }).reset_index()
    prices['price'] = prices['sales_value'] / prices['quantity']
    
    if filter_prices:
        # Уберем слишком дешевые товары.
        low = prices[prices['price'] <= np.nanquantile(prices['price'], q=0.30)]
        low_list = low['item_id'].tolist()
        data  = data.loc[~data['item_id'].isin(low_list)]

        # Уберем слишком дорогие товарыs
        high = prices[prices['price'] > np.nanquantile(prices['price'], q=0.95)]
        high_list = high['item_id'].tolist()
        data = data.loc[~data['item_id'].isin(high_list)]

    # Уберем товары, которые не продавались за последние несколько недель
    if resent_weeks != 0:
        weeks = data.groupby('item_id')['week_no'].last().reset_index()
        weeks = weeks[weeks['week_no'] > (data['week_no'].max() - resent_weeks)]
        sales_items = weeks['item_id'].tolist()
        data = data.loc[data['item_id'].isin(sales_items)]

    # Уберем товары с низкой частотой покупки
    if filter_frequency:
        num_users = data['user_id'].nunique()
        frequency = data.groupby('item_id')['user_id'].nunique().reset_index() 
        frequency.rename(columns={'user_id': 'buyers'}, inplace=True)
        frequency['part_buyers']=frequency['buyers']/num_users
        top_frequency  = frequency[frequency['part_buyers'] > np.nanquantile(frequency['part_buyers'], q=0.30)]\
                        .item_id.tolist()
        data = data.loc[data['item_id'].isin(top_frequency)]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    if popularity_invert:
        data.loc[data['item_id'].isin(top), 'item_id'] = 999999
    else:
        data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    return data



    
def postfilter_items(user_id, recommednations):
    pass