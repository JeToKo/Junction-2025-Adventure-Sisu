import tensorflow
import pandas
import numpy

def neural_network(x_train, y_train, x_test, y_test, lr=1e-3, l2=1e-4, dropout=0.3):
    print("train")
    input_shape = x_train.shape[1]
    reg = tensorflow.keras.regularizers.l2(l2) if l2 else None
    agent = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Input(shape=(input_shape,)),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        tensorflow.keras.layers.Dropout(dropout),
        tensorflow.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Dropout(dropout * 0.5),
        tensorflow.keras.layers.Dense(16, activation='relu', kernel_regularizer=reg),
        tensorflow.keras.layers.Dense(1, activation='sigmoid')
    ])
    agent.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tensorflow.keras.metrics.AUC(name='auc')]
    )
    agent.fit(x_train, y_train, epochs=50, verbose=1, validation_data=(x_test, y_test))

    return agent

def ai_testing(agent, x_test, y_test):
    print("testing")
    accuracy = agent.evaluate(x_test, y_test)
    print("Accuracy: {}".format(accuracy))
    print()

    agent.summary()

    print()

    print("testing")
    correct = 0
    total = 0
    predictions = agent.predict(x_test)
    for i in range(len(y_test)):
        predicted_label = tensorflow.argmax(predictions[i]).numpy()
        actual_label = y_test.iloc[i]
        print('Predicted: {}, Actual: {}'.format(predicted_label, actual_label))
        if predicted_label == actual_label:
            correct += 1
        total += 1
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

def main():
    file = pandas.read_csv('phone_data', names= ['datetime', 'steps', 'screen_time', 'stress', 'migrane'],
                           parse_dates=['datetime'])
    file['year'] = file['datetime'].dt.year
    file['month'] = file['datetime'].dt.month
    file['day'] = file['datetime'].dt.day
    file['hour'] = file['datetime'].dt.hour
    print(file.head())

    x = file[['year', 'month', 'day', 'hour', 'steps', 'screen_time', 'stress']]
    y = file['migrane']

    split_index = int(0.8 * len(file))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]


    agent = neural_network(x_train, y_train, x_test, y_test)
    ai_testing(agent, x_test, y_test)


main()