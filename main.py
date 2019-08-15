import threading
# import self function
import generate as gn
import pvv_model as pvv
#import test as pvv

if __name__ == '__main__':
    print("get...")

    #data = gn.generate_data()
    data = [{"key":"2", "pvv":[[1, 0.1, 0.05, 1],[2, 0.2, 0.1, 3],[3, 0.3, 0.15, 5],[4, 0.4, 0.2, 7],[5, 0.5, 0.25, 9],[6, 0.6, 0.3, 11],[7, 0.7, 0.35, 13],[8, 0.8, 0.4, 15]]}]
    
    pvv_model = pvv.ParameterValueVector(2, 1, 64, 2, 100, 3, 2048)
    pvv_model.pvv_model_train(data)
    
    #data = gn.generate_data()
    data = [{"key":"2", "pvv":[[3, 0.3, 0.15, 5],[4, 0.4, 0.3, 7],[5, 1, 0.25, 10],[6, 0.6, 0.3, 11]]}]
    
    pvv_model.pvv_model_predict(data)
    
