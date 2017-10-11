import pandas as pd


def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


categories_df = pd.read_csv("input/categories.csv", index_col=0)
cat2idx, idx2cat = make_category_tables()

test_pred_df = pd.read_csv('submit/vgg16_full_noAug.01-1.80670-0.68055-0.64533.csv.gz')

grouped = test_pred_df.groupby('_id')
xx = grouped['category'].apply(lambda x: x.mode()[0])

yy = pd.DataFrame({'_id': xx.index, 'category': xx.values})

for index, rows in yy.iterrows():
    rows['category'] = idx2cat[rows['category']]

print("Generating submission file...")
yy.to_csv('submit/testing.csv.gz', index=False, compression='gzip')
