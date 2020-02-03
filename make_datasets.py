import pandas as pd
import pickle

'''
使用了ml-1m数据集

假设一个用户1的数据序列集合为[1,2,3,4,5,6,7,8,9,10,11]
 target_counts = 3, seq_counts = 5, test_counts = 2

则可以生成训练集
   user             seq        target
0     1  [1, 2, 3, 4, 5]    [6, 7, 8]
1     1  [2, 3, 4, 5, 6]    [7, 8, 9]

生成测试机数据

   user             seq        target
0     1  [5, 6, 7, 8, 9]    [10, 11]

'''

def make_datasets(file, target_counts, seq_counts, test_counts,isSave = True):


    #file_path = 'input/u.data'
    file_path = file
    names = ['user', 'item', 'rateing', 'timestamps']
    data = pd.read_csv(file_path, header=None, sep='::', names=names)


    # ReMap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1)))
    item_map[-1] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[-1] = 0
    all_user_count = len(user_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    # Get user session
    data = data.sort_values(by=['user','timestamps']).reset_index(drop=True)

    # 生成用户序列
    user_sessions = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    user_all_items = {}

    for index, row in user_sessions.iterrows():
        user = row['user']
        items = row['item_list']
        user_all_items[user] = items

        # 取每个用户最近的 test_counts 个 交互项目作为测试集合

        for i in range(seq_counts,len(items) - target_counts - test_counts):
            targets = items[i:i+target_counts]
            seqs = items[max(0,i - seq_counts):i]

            train_users.append(user)
            train_seqs.append(seqs)
            train_targets.append(targets)

        #last_item = [items[-1],0,0]
        test_item = items[-1 * test_counts:]
        test_seq =  items[-1 * test_counts - seq_counts:-1 * test_counts]
        test_users.append(user)
        test_seqs.append(test_seq)
        test_targets.append(test_item)


    train = pd.DataFrame({'user':train_users,'seq':train_seqs,'target':train_targets})

    test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    if isSave:

        train.to_csv('input/train.csv', index=False)
        test.to_csv('input/test.csv', index=False)
        with open('input/info.pkl','wb+') as f:
            pickle.dump(user_all_items,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_user_count,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_item_count, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_map, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_map, f, pickle.HIGHEST_PROTOCOL)

    return train,test,\
           user_all_items,all_user_count,\
           all_item_count,user_map,item_map




if __name__ == '__main__':
    make_datasets(3,5,5)
















