import tensorflow
import pandas

def neural_network(x_train, y_train, x_test, y_test):
    print("train")
    input_shape = x_train.shape[1]
    agent = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Input(shape=(input_shape,)),
        tensorflow.keras.layers.Dense(10, activation='relu'),
        tensorflow.keras.layers.Dense(10, activation='relu'),
        tensorflow.keras.layers.Dense(2, activation='softmax')
    ])
    agent.compile(optimizer='adam',
                  loss='mean_squared_error')
    agent.fit(x_train, y_train, epochs=50, batch_size=5, verbose=1)

    return agent

def ai_testing(agent, x_test, y_test):
    print("testing")
    correct = 0
    total = 0
    predictions = agent.predict(x_test)
    for i in range(len(y_test)):
        predicted_label = tensorflow.argmax(predictions[i]).numpy()
        actual_label = y_test.iloc[i]
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