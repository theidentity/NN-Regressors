import numpy as np
from neupy import algorithms
from neupy import storage
import pickle as pkl
from sklearn.metrics import mean_squared_error,r2_score

def load_data():
	trX = np.load('data/npy/trainX.npy')
	trY = np.load('data/npy/trainY.npy')
	teX = np.load('data/npy/testX.npy')
	teY = np.load('data/npy/testY.npy')
	return trX,trY,teX,teY

def save_obj(model,path):
	file = open(path,'wb')
	pkl.dump(model,file)
	print 'Model saved in : '+path

def load_model(path):
	file = open(path,'rb')
	model = pkl.load(file)
	return model

def train(X,y,save_path=None):
	model = algorithms.GRNN(std=0.1,verbose=True)
	model.fit(X,y)
	if save_path is not None:
		save_obj(model,save_path)
	return model

def get_output(model,X):
	y_pred = model.predict(X)
	return y_pred

def get_stats(y_pred,y_true):
	mse = mean_squared_error(y_true,y_pred)
	r2 = r2_score(y_true,y_pred)

	print 'MSE : ',mse
	# print 'r2_score : ',r2

trainX,trainY,testX,testY = load_data()

# Train and save model
# model = train(trainX,trainY,'models/grnn.pkl')

# Load and Test Model
model = load_model('models/grnn.pkl')
print model

y_pred = get_output(model,testX)
get_stats(y_pred,testY)
np.save('data/npy/grnn_test.npy',y_pred)

y_pred_train = get_output(model,trainX)
get_stats(y_pred_train,trainY)
np.save('data/npy/grnn_train.npy',y_pred_train)


