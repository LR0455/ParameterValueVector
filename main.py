import threading
# import self function
import generate as gn
import pvv_model as pvv

if __name__ == '__main__':
    print("get...")

    data = gn.generate_data()
    
    pvv_model = pvv.ParameterValueVector(2, 1, 64, 2, 100, 300, 2048)
    pvv_model.pvv_model_train(data)
    
    data = gn.generate_data()
    pvv_model.pvv_model_predict(data)
    
