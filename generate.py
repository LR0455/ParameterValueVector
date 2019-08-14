import random

def generate_data():
    key_range = 3
    pvv_range = 2
    pvv_vector_range = 10
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

    # print(raw_data)
    
    return raw_data