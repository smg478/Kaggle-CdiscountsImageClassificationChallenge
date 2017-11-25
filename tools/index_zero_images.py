
import numpy as np
import pandas as pd


#train_images_df = pd.read_csv("input/train_images.csv", index_col=0)
#train_zero_df = pd.read_csv("input/train_idx_zero.csv", index_col=0)
test_images_df = pd.read_csv("input/test_images.csv", index_col=0)
x = test_images_df[test_images_df.img_idx == 0]

x.reset_index(drop=True, inplace=True)
#x= pd.DataFrame(x)
x.to_csv("test_idx_zero.csv")