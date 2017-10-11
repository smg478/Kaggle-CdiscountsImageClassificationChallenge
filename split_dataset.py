import pandas as pd
from tqdm import *

# train_images_df  = pd.read_csv("input/train_images.csv", index_col=0)
val_images_df = pd.read_csv("input/val_images.csv", index_col=0)

# split dataset
# train_images_df_50000   = train_images_df[:50000]
# train_images_df_50000.to_csv('input/train_images_50000.csv')

val_images_df_50000 = val_images_df[:50000]
val_images_df_50000.to_csv('input/val_images_50000.csv')

# train_images_df_50_2   = train_images_df[5000000:]
# train_images_df_50_2.to_csv('input/train_images_50_2.csv')

# train_images_df_25_1   = train_images_df[:2500000]
# train_images_df_25_1.to_csv('input/train_images_25_1.csv')

# train_images_df_25_2   = train_images_df[2500000:5000000]
# train_images_df_25_2.to_csv('input/train_images_25_2.csv')

# train_images_df_25_3   = train_images_df[5000000:7500000]
# train_images_df_25_3.to_csv('input/train_images_25_3.csv')

# train_images_df_25_4   = train_images_df[7500000:]
# train_images_df_25_4.to_csv('input/train_images_25_4.csv')

# train_images_df_33_1   = train_images_df[:3300000]
# train_images_df_33_1.to_csv('input/train_images_33_1.csv')

# train_images_df_33_2   = train_images_df[3300000:6600000]
# train_images_df_33_2.to_csv('input/train_images_33_2.csv')

# train_images_df_33_3   = train_images_df[6600000:]
# train_images_df_33_3.to_csv('input/train_images_33_3.csv')
