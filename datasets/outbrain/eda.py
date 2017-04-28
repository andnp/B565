import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt


# ----------------------------
# Generate ad appearance graph
# ----------------------------
# df_train = pd.read_csv('clicks_train.csv')
# df_test = pd.read_csv('clicks_test.csv')
#
# ad_usage_train = df_train.groupby('ad_id')['ad_id'].count()
#
# plt.figure(figsize=(12,6))
# plt.hist(ad_usage_train.values, bins=50, log=True)
# plt.xlabel('Number of times ad appeared', fontsize = 12)
# plt.ylabel('log(Count of displays with ad)', fontsize = 12)
# plt.show()
#
# try:del df_train, df_test
# except:pass; gc.collect()


# -------------------------
# Generate user count graph
# -------------------------
events = pd.read_csv('events.csv')
# print events.head()
#
# uuid_counts = events.groupby('uuid')['uuid'].count()
#
# plt.figure(figsize=(12, 4))
# plt.hist(uuid_counts.values, bins=50, log=True)
# plt.xlabel('Number of times user appeared in set', fontsize=12)
# plt.ylabel('log(Count of users)', fontsize=12)
# plt.show()
#
# try:del events
# except:pass; gc.collect()

# ---------------------------
# Filter topics by confidence
# ---------------------------

topics = pd.read_csv('documents_topics.csv')
# topic_ids_count = topics.loc[topics['confidence_level'] > .1].count()
topic_ids = np.sort(pd.Series.unique(topics['topic_id']))
doc_ids_events = np.sort(pd.Series.unique(events['document_id']))
total_document_ids_events = len(doc_ids_events)
total_topic_ids = len(topic_ids)

m_size = (total_document_ids_events, total_topic_ids)
m = np.zeros(m_size)

visited_dict = {}
i = 0
missing_topics = 0
for doc_id in doc_ids_events:
    row = topics[topics['document_id'] == doc_id]
    if len(row) == 0:
        missing_topics += 1
        continue
    t_id = row['topic_id'].values[0]
    confidence = row['confidence_level'].values[0]
    if visited_dict.get(doc_id, -1) == -1: # new doc_id
        visited_dict[doc_id] = i
        m[i][t_id] = confidence
    else:
        prev_row = visited_dict[doc_id]
        m[prev_row][t_id] = confidence
    i += 1
print missing_topics
df = pd.DataFrame(data=m.astype(float))
df.to_csv('outfile.csv', sep=',', header=False, float_format='%.4f', index=False)

# Transform PCA
# ref: http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
def PCA(data, dims_rescaled_data=2):
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs
