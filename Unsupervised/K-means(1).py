# FIRST CLUSTERING ATTEMPT WITHOUT PRIOR TRIES

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/milaan9/Clustering-Datasets/master/01.%20UCI/banknote.csv")

y = df['class']
X = df.drop(['class'], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

from sklearn.cluster import KMeans

KM = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, random_state=0)
KM.fit(X_train)
print(KM.labels_)

#PLOT