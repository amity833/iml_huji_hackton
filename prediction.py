import statistics

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

positive_sign = ['extensive', 'yes', '(+)', 'ye', 'Ye', 'po', 'PO', 'Po', 'os',
                 'high', 'High', 'HIGH', '100']
negative_sign = ['No', '(-)', 'NO', 'no', 'NE', 'Ne', 'ne', 'eg', 'ng', 'Ng',
                 'NG', "שלילי", 'Low', 'low', 'LOW']

indeterminate_sign = ['בינוני', "Inter", "Indeter", "indeter", "inter"]


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    pd.DataFrame(estimator.predict(X),
                 columns=["predicted_values"]).to_csv(filename, index=False)


def feature_evaluation(X: pd.DataFrame, y: pd.Series):
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_axis = []

    y_np = y.to_numpy()
    y_np = y_np.ravel()

    for i, feature in enumerate(X.columns):
        y_axis.append(
            np.cov(X[feature], y_np)[0, 1] / (
                    np.std(X[feature]) * np.std(y_np)))

    for i, feature in enumerate(X):
        create_scatter_for_feature(X[feature], y_np, round(y_axis[i], 3),
                                   feature)


def create_scatter_for_feature(X: pd.DataFrame, y: np.array, title,
                               feature_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers"))
    fig.update_layout(title="Pirson: " + str(title),
                      xaxis_title=feature_name,
                      yaxis_title="y")

    # fig.show()


def her_2_pre(x):
    global positive_sign, negative_sign, indeterminate_sign
    pos_lst = positive_sign + indeterminate_sign + ["2,3", "+"]
    neg_lst = negative_sign + ["0", "1", "-"]
    x = str(x)
    for p in pos_lst:
        if p in x:
            return "pos"

    for n in neg_lst:
        if n in x:
            return "neg"

    return "null"


def er_pr_pre(x):
    global positive_sign, negative_sign, indeterminate_sign
    pos_lst = positive_sign + indeterminate_sign + ["+", "3", "4", "90", "80",
                                                    "70"]
    neg_lst = negative_sign + ["-"]
    x = str(x)
    for p in pos_lst:
        if p in x:
            return "pos"
    for n in neg_lst:
        if n in x:
            return "neg"
    return "null"


def KI67_score(x):
    if 0 < x <= 5:
        return 1
    if 5 < x < 10:
        return 2
    if 10 <= x < 50:
        return 3
    if 50 <= x <= 100:
        return 4
    else:
        return 0


def KI67_pre(x):
    for sub in ['Sc', 'sc']:
        out = x.find(sub)
        if out != -1:
            for i in range(out, len(x)):
                if x[i].isdigit():
                    return int(x[i])
    sep = ['-', ' ', '=']
    x = "".join(filter(lambda c: c in sep or c.isdigit(), x)).replace('-',
                                                                      ' ').replace(
        '=', ' ')
    x = [int(s) for s in x.split(' ') if s.isdigit()]
    if x == []:
        return 0
    x = statistics.mean(x)
    if 0 < x < 100:
        return x
    else:
        return 0


def Histological_diagnosis_pre(x):
    d = {"INFILTRATING DUCT CARCINOMA": "IDC",
         "LOBULAR INFILTRATING CARCINOMA": "LIC",
         "INTRADUCTAL CARCINOMA": "IC",
         "INFILTRATING DUCTULAR CARCINOMA WITH DCIS": "IDCWD"}

    if x in d:
        return d[x]
    else:
        return "non common"


def how_much_per_unique(x, d: dict):
    if x in d:
        d[x] += 1
    else:
        d[x] = 1


def Lymphatic_penetration_pre(x):
    if x[:2] == "L0":
        return '0'
    elif x[:2] == "L1" or x[:2] == "LI":
        return 1
    elif x[:2] == "L2":
        return '2'
    return 'null'


def Lymphovascular_invasion_pre(x):
    x = str(x)
    global positive_sign, negative_sign, indeterminate_sign
    lst_of_pos = positive_sign + ["+"]
    lst_of_neg = negative_sign + ["-"]
    very_danger = 'MICROPAPILLARY VARIANT'

    if x == very_danger:
        return 'very danger'

    for p in lst_of_pos:
        if p in x:
            return 'p'

    for n in lst_of_neg:
        if n in x:
            return 'n'

    return 'null'


def find_score(x: str):
    for sub in ['Sc', 'sc']:
        out = x.find(sub)

        if out != -1:
            for i in range(out, len(x)):
                if x[i].isdigit():
                    return x[i]
                elif x[i] == 'I':
                    return 1
    return None


def metastases_mark_pre(x):
    d = {"M1": "1", "M1a": "1", "M1b": "1", "M0": "0"}
    if x in d:
        return d[x]
    else:
        return "null"


def Tumor_mark_pre(x):
    d = {'T2': 2, 'T4': 4, 'T1c': 1, 'T1b': 1, 'MF': 0,
         'T1': 1, 'Tis': 1061, 'T1mic': 1, 'Tx': 0, 'T3': 3, 'T1a': 1,
         'Not yet Established': 0, 'T0': 0, 'T3c': 3, 'T2a': 2, 'T4d': 4,
         'T4c': 4, 'T4a': 4, 'T3b': 3, 'T2b': 2, 'T4b': 4, 'T3d': 3}
    if x in d:
        return d[x]
    else:
        return 0


def Stage_pre(x):
    if type(x) == str:
        x = "".join(filter(lambda c: c.isdigit(), x))
    if x in ["0", "1", "2", "3", "4"]:
        return x
    return "null"


def side(x):
    if x['both']:
        x['l'] = 1
        x['r'] = 1


def preprocess(df: pd.DataFrame):
    # Histological diagnosis
    df["Histological diagnosis"] = df["Histological diagnosis"].apply(
        lambda x: Histological_diagnosis_pre(x))

    # Ivi -Lymphovascular invasion
    df["Ivi -Lymphovascular invasion"] = df[
        "Ivi -Lymphovascular invasion"].apply(
        lambda x: Lymphovascular_invasion_pre(x))

    # Lymphatic penetration
    df["Lymphatic penetration"] = df["Lymphatic penetration"].apply(
        lambda x: Lymphatic_penetration_pre(x))

    # cur_date
    today = datetime.strptime("6/2/2022", '%d/%m/%Y')
    df["Diagnosis date"] = df["Diagnosis date"].apply(
        lambda x: (datetime.strptime(x[:10], '%d/%m/%Y') - today).days * -1)

    # Her2 preprocessing
    df["Her2"] = df["Her2"].apply(lambda x: her_2_pre(x))

    # M -metastases mark (TNM)
    df["M -metastases mark (TNM)"] = df["M -metastases mark (TNM)"].apply(
        lambda x: metastases_mark_pre(x))

    # er
    df["er"] = df["er"].apply(lambda x: er_pr_pre(x))
    df["pr"] = df["pr"].apply(lambda x: er_pr_pre(x))

    # Stage
    df["Stage"] = df["Stage"].apply(lambda x: Stage_pre(x))

    # make categorical
    X = pd.get_dummies(df, columns=[" Hospital",
                                    " Form Name",
                                    "Histopatological degree",
                                    "Ivi -Lymphovascular invasion",
                                    "Histological diagnosis",
                                    "M -metastases mark (TNM)",
                                    "Lymphatic penetration",
                                    "Her2",
                                    "er",
                                    "pr",
                                    "Stage"])

    # # Side
    # redundant_dummy = pd.get_dummies(X["Side"])
    # redundant_dummy.rename(
    #     columns={"ימין": "r", "שמאל": "l", "דו צדדי": "both"}, inplace=True)
    # redundant_dummy.apply(side, axis=1)
    #
    # del X["Side"]
    # X = pd.concat([redundant_dummy[["l", "r"]], X])

    # Age  preprocessing
    X = X[X["Age"] < 120]
    X = X[0 < X["Age"]]

    # Basic stage preprocessing
    X["Basic stage"] = X["Basic stage"].replace(
        {'Null': 0, 'c - Clinical': 1, 'p - Pathological': 2,
         'r - Reccurent': 3})

    # KI67 protein preprocessing
    # print(sum(X["KI67 protein"].apply(lambda x: 1 if validate(x) else 0)))
    X["KI67 protein"] = X["KI67 protein"].astype(str)
    X["KI67 protein"] = X["KI67 protein"].apply(
        lambda x: KI67_score(KI67_pre(x)))

    # margin type
    margin_neg = ['נקיים', 'ללא']
    margin_pos = ['נגועים']
    X["Margin Type"] = X["Margin Type"].apply(
        lambda x: 1 if x in margin_pos else 0)

    # Nodes exam pre_processing
    X["Nodes exam"] = X["Nodes exam"].fillna(0)

    # Positive nodes
    X["Positive nodes"] = X["Positive nodes"].fillna(0)

    # T -Tumor mark (TNM)
    X["T -Tumor mark (TNM)"] = X["T -Tumor mark (TNM)"].apply(
        lambda x: Tumor_mark_pre(x))

    X.drop(
        ['Side', 'N -lymph nodes mark (TNM)', 'Tumor depth', 'Tumor width',
         'User Name', 'Surgery date1', 'Surgery date2',
         'Surgery date3', 'Surgery name1', 'Surgery name2', 'Surgery name3',
         'Surgery sum', 'surgery before or after-Activity date',
         'surgery before or after-Actual activity',
         'id-hushed_internalpatientid'], axis=1, inplace=True)

    return X


if __name__ == '__main__':
    np.random.seed(0)

    # Load data and preprocess

    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
    full_data.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)

    # data_path, y_location_of_distal, y_tumor_path = sys.argv[1:]

    original_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")

    # remove heb prefix
    original_data.rename(columns=lambda x: x.replace('אבחנה-', ''),
                         inplace=True)

    y_tumor = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.1.csv")
    y_tumor.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)

    print({f: original_data[f].unique().size for f in original_data.columns})
    print()

    d = {}

    # original_data["Surgery sum"].apply(
    #     lambda x: how_much_per_unique(x, d))
    # print(d)

    X = preprocess(original_data)

    # feature_evaluation(X[["Age", "Her2", "Basic stage"]], y_tumor)
    print("this is me")
