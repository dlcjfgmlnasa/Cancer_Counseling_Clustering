# -*- coding:utf-8 -*-
import os
import argparse
import pandas as pd
from data_helper import tf_idf
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser('Cancer_Counseling_Clustering')
    parser.add_argument(
        '--n_clusters',
        default=15,
        type=int
    )
    parser.add_argument(
        '--max_iter',
        default=100,
        type=int
    )
    parser.add_argument(
        '--model_path',
        default='./cluster.pkl',
        type=str
    )

    return parser.parse_args()


def train_mode(x_data, n_clusters, max_iter, filename):
    # train model
    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter)
    model.fit(x_data)
    joblib.dump(model, filename)
    return model


def main():
    args = get_args()   # get arguments

    # load data ( data => pos tagging -> filter pos -> tf_idf )
    x_data = tf_idf()

    # train model or load model
    if not os.path.exists(args.model_path):
        model = train_mode(x_data, args.n_clusters, args.max_iter, args.model_path)
    else:
        model = joblib.load(args.model_path)

    # show cluster data size
    plt.hist(model.labels_, bins=args.n_clusters)
    plt.show()

    # assignments
    cluster_assignments_dict = {i: [] for i in range(args.n_clusters)}
    base_lines = open('./data/combine_cancer_counseling_base.txt', encoding='utf-8').readlines()

    for i, cluster in enumerate(model.predict(x_data)):
        cluster_assignments_dict[cluster].append(base_lines[i].strip())
        # print('cluster : {0} contents : {1}'.format(cluster, base_lines[i].strip()))

    # save file (./data/combine_cancer_counseling_cluster.xlsx)
    with pd.ExcelWriter('./data/combine_cancer_counseling_cluster.xlsx') as writer:
        for key in cluster_assignments_dict.keys():
            sheet_name = 'cluster_{}'.format(key)
            sample = pd.DataFrame(
                [line.split('<SPLIT>') for line in cluster_assignments_dict[key]],
                columns=['title', 'content']
            )
            sample.to_excel(writer, sheet_name=sheet_name)


if __name__ == '__main__':
    main()
