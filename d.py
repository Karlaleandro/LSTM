import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#apple_training_complete = pd.read_csv('esse2.csv')
apple_training_complete = pd.read_csv('pet.csv')

apple_training_processed = apple_training_complete.iloc[:, 1:2].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

apple_training_scaled = scaler.fit_transform(apple_training_processed)

features_set = []
labels = []
#for i in range(60, 1260):
for i in range(60, 1243):
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(features_set, labels, epochs = 100, batch_size = 32)

apple_testing_complete = pd.read_csv('test.csv')
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values
#
apple_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)
#
test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
#
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
#
test_features = []
for i in range(60, 80):
    test_features.append(test_inputs[i-60:i, 0])

#
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
#

predictions = model.predict(test_features)
#
predictions = scaler.inverse_transform(predictions)
#
'''
gridsize = (3, 2)
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(gridsize, (2, 0))
ax3 = plt.subplot2grid(gridsize, (2, 1))
c = list(range(1,1261))
ax1.set_title('Home value as a function of home age & area population',fontsize=14)
sctr = ax1.scatter(x=apple_testing_processed, y= c,cmap='RdYlGn')

plt.colorbar(sctr, ax=ax1, format='$%d')
ax1.set_yscale('log')
ax2.hist(age, bins='auto')
ax3.hist(pop, bins='auto', log=True)


add_titlebox ( ax2 ,  'Histograma: idade home' ) 
add_titlebox ( ax3 ,  'Histograma: população da área (SCL log).' )


plt.figure(figsize=(10,6))
plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')
plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
'''