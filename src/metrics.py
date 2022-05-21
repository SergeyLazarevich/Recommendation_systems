"""
Metrics

"""
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (precision_score, f1_score, accuracy_score, \
                             classification_report, \
                             recall_score, confusion_matrix, roc_curve, \
                             precision_recall_curve, ConfusionMatrixDisplay, \
                             RocCurveDisplay,PrecisionRecallDisplay, roc_auc_score)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    if (type(recommended_list) != float) and (type(bought_list) != float):
        return hit_rate(recommended_list[:k], bought_list)
    else: return 0

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    if (type(recommended_list) != float) and (type(bought_list) != float):
        return precision(recommended_list[:k], bought_list)
    else: return 0

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    if (type(recommended_list) != float) and (type(bought_list) != float):
        return recall(recommended_list[:k], bought_list)
    else: return 0


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[recommended_list <= k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)


    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant

def calc_get_own(df_data, recommend_model, columns_name, top_k, USER_COL='user_id'):
    df_data[columns_name] = df_data[USER_COL].apply(lambda x: recommend_model(x, N=top_k))
    return df_data

def evalRecall(df_data, recommend_model, top_k, n_predict, USER_COL='user_id', ACTUAL_COL= 'actual'):
    for name, model in recommend_model.items():
        df_data = calc_get_own(df_data, model, name, n_predict, USER_COL)
    recall = sorted(calc_recall(df_data, top_k, ACTUAL_COL), key=lambda x: x[1],reverse=True)
    return pd.DataFrame(recall, columns=['Name', 'Score'])

def evalPrecision(df_data, recommend_model, top_k, n_predict, USER_COL='user_id', ACTUAL_COL= 'actual'):
    for name, model in recommend_model.items():
        df_data = calc_get_own(df_data, model, name, n_predict, USER_COL)
    precision = sorted(calc_precision(df_data, top_k, ACTUAL_COL), key=lambda x: x[1],reverse=True)
    return pd.DataFrame(precision, columns=['Name', 'Score'])

def calc_recall(df_data, top_k, ACTUAL_COL= 'actual'):
    for col_name in df_data.columns[1:]:
        yield col_name, round(df_data.apply(lambda row: recall_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean(), 3)

def calc_precision(df_data, top_k, ACTUAL_COL= 'actual'):
    for col_name in df_data.columns[1:]:
        yield col_name, round(df_data.apply(lambda row: precision_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean(), 3)

def rerank(user_id, df, TOPK_PRECISION, USER_COL='user_id'):
    return df[df[USER_COL]==user_id].sort_values('proba_item_purchase', ascending=False).head(TOPK_PRECISION).item_id.tolist()

def print_stats_data(df_data, name_df, USER_COL='user_id', ITEM_COL='item_id'):
    print(name_df)
    print(f"Shape: {df_data.shape} Users: {df_data[USER_COL].nunique()} Items: {df_data[ITEM_COL].nunique()}\n")


def show_proba_calibration_plots(y_predicted_probs, y_true_labels):
    
    thresholds = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in np.arange(0.01, 1, 0.01):
        thresholds.append(threshold)
        precisions.append(precision_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))
        recalls.append(recall_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))
        f1_scores.append(f1_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))

    scores_table = pd.DataFrame({'f1':f1_scores,
                                 'precision':precisions,
                                 'recall':recalls,
                                 'tresholds':thresholds}).sort_values('f1', ascending=False).round(3)
  
    figure = plt.figure(figsize = (15, 5))

    plt1 = figure.add_subplot(121)
    plt1.plot(thresholds, precisions, label='Precision', linewidth=4)
    plt1.plot(thresholds, recalls, label='Recall', linewidth=4)
    plt1.plot(thresholds, f1_scores, label='F1', linewidth=4)
    plt1.set_ylabel('Scores')
    plt1.set_xlabel('Probability threshold')
    plt1.set_title('Probabilities threshold calibration')
    plt1.legend(bbox_to_anchor=(0.25, 0.25))   
    plt1.table(cellText = scores_table.values[:10, :],
               colLabels = scores_table.columns, 
               colLoc = 'center', cellLoc = 'center', loc = 'bottom', bbox = [0, -1.3, 1, 1])                                                                               

    
    preds_with_true_labels = np.array(list(zip(y_predicted_probs, y_true_labels)))
    
    plt2 = figure.add_subplot(122)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 0][:, 0], 
              label='Another class', color='royalblue', alpha=1, bins=15)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 1][:, 0], 
              label='Main class', color='darkcyan', alpha=0.8, bins=15)
    plt2.set_ylabel('Number of examples')
    plt2.set_xlabel('Probabilities')
    plt2.set_title('Probability histogram')
    plt2.legend(bbox_to_anchor=(1, 1))

    plt.show()

    return np.round(roc_auc_score(y_true_labels, y_predicted_probs), 3), \
            scores_table.values[0, 0], scores_table.values[0, 1], \
            scores_table.values[0, 2], scores_table.values[0, 3]

#--------------------------------------------------------------------------------
# Function to plot a Confusion Matrix
def Plot_Confusion_Matrix(CM,Title="Confusion Matrix"):
    GroupNames       = ["True Neg","False Pos","False Neg","True Pos"]
    GroupCounts      = ["{0:0.0f}".format(x) for x in CM.flatten()]
    GroupPercentages = ["{0:.2%}".format(x) for x in CM.flatten()/np.sum(CM)]
    
    Labels = [f"{x1}\n\n{x2}\n\n{x3}" for x1,x2,x3 in zip(GroupNames,GroupCounts,GroupPercentages)]
    Labels = np.asarray(Labels).reshape(2,2)
    
    FontSize = 12
    plt.figure(figsize=(6,5))
    Ax = sns.heatmap(CM, annot=Labels, fmt="", cmap="Blues")
    Ax.set_title(Title, fontsize=FontSize)
    Ax.set_xlabel("Predicted Values", fontsize=FontSize)
    Ax.set_ylabel("Actual Values", fontsize=FontSize)
    Ax.xaxis.set_ticklabels(["False","True"], fontsize=FontSize)
    Ax.yaxis.set_ticklabels(["False","True"], fontsize=FontSize)
    plt.show()
    return


def evaluate_preds(y_pred, y_test, tresholds):
    probs = y_pred >= tresholds
    print(classification_report(y_test, probs))
    Plot_Confusion_Matrix(confusion_matrix(y_test, probs))
    RocCurveDisplay.from_predictions(y_test, y_pred)
    PrecisionRecallDisplay.from_predictions(y_test, y_pred)
    plt.show()


