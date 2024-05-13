import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.layers import Input #type: ignore
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_squared_error



train_df = pd.read_csv(r'C:\Users\User\Desktop\UNI\Delta\DELTA\Train_Time_Series.csv')
test_df = pd.read_csv(r'C:\Users\User\Desktop\UNI\Delta\DELTA\Test_Time_Series.csv')
train_df = train_df.set_index('Date')
test_df=test_df.set_index('Date')
plt.figure(figsize=(10, 6))
plt.plot(train_df['Trading Volume (Target Variable)'])
plt.plot(test_df['10'])
plt.show()

scaler = StandardScaler() 
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.fit_transform(test_df)





train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
test_scaled_df = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)
plt.figure(figsize=(10, 6))
plt.plot(train_scaled[321])
plt.plot(train_scaled[0])
plt.show()
#print(train_df.shape)
X = train_scaled_df.drop('Trading Volume (Target Variable)', axis=1)
y = train_scaled_df['Trading Volume (Target Variable)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print (X_train.shape)
print(y_train.shape) 
plt.show()



print(X_train.shape[1]) 
print(X_test.shape[1]) 

model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Use Input layer to define input shape
    Dense(64, activation='sigmoid', name='layer1'),
    Dense(32, activation='sigmoid', name='layer2'),
    Dense(1)
])

epochs = 30
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = keras.optimizers.SGD(learning_rate=learning_rate, decay=decay_rate, nesterov=False, momentum=momentum)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mean_squared_error'])# batch_size = 56 #1 for online SGD
model.summary()
story = model.fit(X_train, y_train, #name model differently
                    batch_size=56,
                    epochs=epochs*2,
                    verbose=0,
                 validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

# trainPredict= model.predict(X_train)
# #print(trainPredict)
# testPredict = model.predict(X_test)
# #print(testPredict)
# trainScore = math.sqrt(mean_squared_error(y_train, trainPredict[:,0]))
# print('Train Score: %.10f RMSE' % (trainScore))

# testScore = math.sqrt(mean_squared_error(y_test, testPredict[:,0]))
# print('Test Score: %.10f RMSE' % (testScore))


# # opt = keras.optimizers.SGD(learning_rate=0.1) # there are many more parameters that we could be adding here

# # # Calling compile and specifying some mandatory arguments completes setting up the NN

# epochs = 30
# batch_size = 56
# # #And now we fit the model (that is the training part), mind that saving it to the 
# # # variable will help you retrieve and analyse the training history
# story = model.fit(X_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(X_test, y_test) #OR validation_split=0.3
#                  ) 
# score = model.evaluate(X_test, y_test, verbose=0)
# print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

story.history.keys()
def show_history(story):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(story.history['loss'], label='Training Loss')
    ax.plot(story.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Loss during Training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    plt.show()

show_history(story)