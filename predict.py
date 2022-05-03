import pickle

def predict():
    file_to_read = open('model.pickle', "rb")
    model = pickle.load(file_to_read)
    file_to_read.close()
    class_probabilities = model.predict([[3, 3, 2, 2]])
    return class_probabilities[0]

if __name__ == '__main__':
    predict()
