import random

def key_merge(all_data):
    merge_data = {}
    for data in all_data:
        key = data['key'] 
        pvv = data['pvv']
        if key not in merge_data:
            merge_data[key] = pvv
        else:
            merge_data[key] += pvv
    
    # print(merge_data)
    return merge_data
    
def generate_data():
    key_range = 2
    pvv_range = 5
    pvv_vector_range = 5
    pvv_value_range = 20
    id_range = 5000

    raw_data = []

    
    pvv_vectors_len = [0, 3, 1, 2, 4, 3]
    # pvv_vectors_len = [0]

    # for i in range(key_range):
    #     pvv_vectors_len.append(random.randint(5, pvv_vector_range))

    for i in range(20):
        data = {}
        data['id'] = random.randint(1, id_range)
        data['key'] = random.randint(1, key_range)
        data['pvv'] = []

        pvv_len = random.randint(1, pvv_range)
        pvv_vector_len = pvv_vectors_len[data['key']]

        for j in range(pvv_len):
            pvv = []
            for k in range(pvv_vector_len):
                pvv.append(random.randint(1, pvv_value_range))
            data['pvv'].append(pvv)

            raw_data.append(data)

    print(raw_data)
    
    return raw_data