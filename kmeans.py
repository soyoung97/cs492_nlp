import pandas as pd
from sklearn.cluster import KMeans

sample_td_matrix = [[0.00000e+00,  5.11364e-03,  1.19136e-02,  2.73889e-03],  [2.51731e-03,  9.37207e-04,  4.33369e-03,  1.14943e-03], [1.19048e-02,  1.97239e-03,  3.40136e-03,  1.82899e-03],  [2.09644e-03,  5.63380e-03,  0.00000e+00,  1.81598e-03],  [1.09290e-02,  0.00000e+00,  9.93542e-04,  2.86807e-03],  [6.21118e-03,  1.84502e-03,  2.85714e-03,  0.00000e+00],  [0.00000e+00,  2.34009e-03,  0.00000e+00,  9.28793e-03]]
doctitle = ['lead', 'receive', 'play', 'american']
df = pd.DataFrame(sample_td_matrix, columns=doctitle)
num_clusters = 3 # you can change number of clusters here
kmeans = KMeans(n_clusters=num_clusters).fit(sample_td_matrix)
df['category'] = kmeans.labels_
print(df)
