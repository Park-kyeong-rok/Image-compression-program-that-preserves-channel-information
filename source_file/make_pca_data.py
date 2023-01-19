from argument import args
import numpy as np
from data_utils import data_split
from sklearn.decomposition import PCA

#데이터는 .npy 형식이며, 모양은 (데이터 갯수, 높이, 넢이, 채널수)입니다.
data_x = np.load(f'data/{args.data_name}_x.npy')
data_y = np.load(f'data/{args.data_name}_y.npy')

#데이터를 6:2:2로 나누어줍니다.
train_x, val_x, test_x, train_y, test_y, val_y = data_split(data_x, data_y)

#각 채널 안에서 높이와 넓이 정보를 취합합니다. 모양은 (데이터 갯수, 높이x넓이, 채널수)
train_x = train_x.reshape(train_x.shape[0],-1,3)
val_x = val_x.reshape(val_x.shape[0],-1,3)
test_x = test_x.reshape(test_x.shape[0],-1,3)

#PCA시 각 채널의 정보를 모두 기억할 수 있도록 채널 별로 pca를 적용해줍니다.
pca_r = PCA(n_components=args.pca_d)
pca_g = PCA(n_components=args.pca_d)
pca_b = PCA(n_components=args.pca_d)

rgb_list = [pca_r, pca_g, pca_b]

train_list = []
val_list = []
test_list = []

train_list.append(pca_r.fit_transform(train_x[:,:,0]))
train_list.append(pca_g.fit_transform(train_x[:,:,1]))
train_list.append(pca_b.fit_transform(train_x[:,:,2]))
#최종적으로 데이터 모양은 (데이터 갯수, PCA 차원 * 채널수)
train_x = np.concatenate((train_list[0], train_list[1], train_list[2]), axis = 1)
np.save(f'processed_data/{args.data_name}_train_x.npy', train_x)
np.save(f'processed_data/{args.data_name}_train_y.npy', train_y)

val_list.append(pca_r.transform(val_x[:,:,0]))
val_list.append(pca_g.transform(val_x[:,:,1]))
val_list.append(pca_b.transform(val_x[:,:,2]))
val_x = np.concatenate((val_list[0], val_list[1], val_list[2]), axis = 1)
np.save(f'processed_data/{args.data_name}_val_x.npy', val_x)
np.save(f'processed_data/{args.data_name}_val_y.npy', val_y)

test_list.append(pca_r.transform(test_x[:,:,0]))
test_list.append(pca_g.transform(test_x[:,:,1]))
test_list.append(pca_b.transform(test_x[:,:,2]))
test_x = np.concatenate((test_list[0], test_list[1], test_list[2]), axis = 1)
np.save(f'processed_data/{args.data_name}_test_x.npy', test_x)
np.save(f'processed_data/{args.data_name}_test_y.npy', test_y)