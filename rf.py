# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def binary(x):
    """
    transform data to binary
    :param x:
    :return:
    """
    if (x is True) or (x == 'True'):
        return 1
    elif (x is False) or (x == 'False'):
        return 0


def category(x):
    """
    transform data to category
    :param x:
    :return:
    """
    if x == 0:
        return 'functional'
    elif x == 1: 
        return 'functional needs repair'
    elif x == 2:
        return 'non functional'


def main():
    """
    random forest
    :return:
    """
    # load data from csv
    train_values = pd.read_csv('./input/training_set_values.csv')
    train_labels = pd.read_csv('./input/training_set_labels.csv')
    test_values = pd.read_csv('./input/test_set_values.csv')
    # merge data
    train_data = pd.concat([train_values, test_values])

    # remove useless features
    train_data.drop(['recorded_by'], axis=1, inplace=True)
    train_data.drop(['region'], axis=1, inplace=True)
    train_data.drop(['extraction_type', 'extraction_type_group'], axis=1, inplace=True)
    train_data.drop(['management'], axis=1, inplace=True)
    train_data.drop(['payment'], axis=1, inplace=True)
    train_data.drop(['water_quality'], axis=1, inplace=True)
    train_data.drop(['quantity'], axis=1, inplace=True)
    train_data.drop(['source', 'source_class'], axis=1, inplace=True)
    train_data.drop(['waterpoint_type'], axis=1, inplace=True)
    # encode
    train_labels['status_group'] = train_labels['status_group'].astype('category')
    train_labels['status_group'] = train_labels['status_group'].cat.codes

    # encode and fill na with -1
    train_data['public_meeting'] = train_data['public_meeting'].apply(binary)
    train_data['public_meeting'].fillna(-1, inplace=True)
    train_data['permit'] = train_data['permit'].apply(binary)
    train_data['permit'].fillna(-1, inplace=True)

    # turn date to year
    train_data.date_recorded = train_data.date_recorded.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    train_data['recorded_year'] = train_data['date_recorded'].dt.year
    # operational_years = recorded_year - construction_year
    op_years = list(train_data.recorded_year - train_data.construction_year)
    operational_years = []
    for i in op_years:
        if (i > 500) or (i < 0):
            operational_years.append(0)
        else:
            operational_years.append(i)

    train_data['operational_years'] = operational_years
    # remove useless features
    train_data.drop(['recorded_year'], axis=1, inplace=True)
    train_data.drop(['construction_year'], axis=1, inplace=True)
    train_data.drop(['date_recorded'], axis=1, inplace=True)

    # encode
    train_data['waterpoint_type_group'] = train_data['waterpoint_type_group'].astype('category')
    train_data['waterpoint_type_group'] = train_data['waterpoint_type_group'].cat.codes
    train_data['source_type'] = train_data['source_type'].astype('category')
    train_data['source_type'] = train_data['source_type'].cat.codes
    train_data['quantity_group'] = train_data['quantity_group'].astype('category')
    train_data['quantity_group'] = train_data['quantity_group'].cat.codes
    train_data['quality_group'] = train_data['quality_group'].astype('category')
    train_data['quality_group'] = train_data['quality_group'].cat.codes
    train_data['payment_type'] = train_data['payment_type'].astype('category')
    train_data['payment_type'] = train_data['payment_type'].cat.codes
    train_data['management_group'] = train_data['management_group'].astype('category')
    train_data['management_group'] = train_data['management_group'].cat.codes
    train_data['extraction_type_class'] = train_data['extraction_type_class'].astype('category')
    train_data['extraction_type_class'] = train_data['extraction_type_class'].cat.codes
    train_data['basin'] = train_data['basin'].astype('category')
    train_data['basin'] = train_data['basin'].cat.codes
    train_data['scheme_management'] = train_data['scheme_management'].astype('category')
    train_data['scheme_management'] = train_data['scheme_management'].cat.codes
    train_data['funder'] = train_data['funder'].astype('category')
    train_data['funder'] = train_data['funder'].cat.codes
    train_data['installer'] = train_data['installer'].astype('category')
    train_data['installer'] = train_data['installer'].cat.codes
    train_data['wpt_name'] = train_data['wpt_name'].astype('category')
    train_data['wpt_name'] = train_data['wpt_name'].cat.codes
    train_data['lga'] = train_data['lga'].astype('category')
    train_data['lga'] = train_data['lga'].cat.codes
    train_data['ward'] = train_data['ward'].astype('category')
    train_data['ward'] = train_data['ward'].cat.codes
    train_data['scheme_name'] = train_data['scheme_name'].astype('category')
    train_data['scheme_name'] = train_data['scheme_name'].cat.codes
    train_data['subvillage'] = train_data['subvillage'].astype('category')
    train_data['subvillage'] = train_data['subvillage'].cat.codes

    # remove na values
    train_data = train_data.dropna()

    # define classifier
    rf = RandomForestClassifier(n_estimators=100)

    # split data
    train_label = train_labels['status_group']
    train_feature, test_data = train_data[:59400], train_data[59400:]
    x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2, random_state=1)

    # fit model with training dataset
    rf.fit(x_train, y_train)

    # predict test dataset and output accuracy score
    accuracy = accuracy_score(rf.predict(x_test), y_test)
    # about 0.82
    print("Accuracy = " + str(accuracy))

    # predict
    prediction = rf.predict(test_data)
    # submission dataframe
    submission = pd.DataFrame({'id': test_data['id'], 'status_group': prediction})
    submission['status_group'] = submission['status_group'].apply(category)
    # output dataframe to csv file
    submission.to_csv("./output/submission.csv", index=False)


if __name__ == '__main__':
    main()
