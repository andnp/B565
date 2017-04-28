import pandas as pd

start = 100000
name = 'small_train2.csv'

m = pd.read_csv('merge.csv')
m = m.drop('Unnamed: 0', axis=1)

train = pd.read_csv('clicks_train.csv')

p = pd.read_csv('promoted_content.csv')
p = p.drop('campaign_id', axis=1)
p = p.drop('advertiser_id', axis=1)

e = pd.read_csv('events.csv')
e = e.drop('uuid', axis=1)
e = e.drop('timestamp', axis=1)
e = e.drop('platform', axis=1)
e = e.drop('geo_location', axis=1)

small = train[start: start + 100000]

small = small.merge(e, on='display_id')
small = small.drop('display_id', axis=1)
small = small.merge(m, on='document_id')
small = small.drop('document_id', axis=1)

small = small.merge(p, on='ad_id')
small = small.drop('ad_id', axis=1)
small = small.merge(m, on='document_id')
small = small.drop('document_id', axis=1)

small.to_csv(name, index=False)
