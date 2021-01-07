import pandas as pd
import pickle as pkl

def pkl_df(file_path, path2save):
    # Load the entity dictionary
    
    e2t = dict()
    with open(file_path, 'r+') as f:
        for line in f.readlines():
            pair = line.split('\t', 1)
            value = pair[1].replace('\n', '')
            e2t[pair[0]] = value



    # entity2text = pd.read_csv(file_path, header=None, sep='\t')
    # entity2text.columns = ['a', 'b']
    # entity_dict = entity2text.set_index('Entity').T.to_dict('series')
    # del entity2text

    with open(path2save+'.pkl', 'wb') as f:
        pkl.dump(e2t, f)
    print('Dataframe converted to dictionary and saved here: {}'.format(path2save))

if __name__ == "__main__":
    path = 'data/fb15k/'
    txt_file = 'entity2type.txt'
    pkl_file = 'e2type_dict'
    pkl_df(path+txt_file, path+pkl_file)
