# python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_binary_classifier(input_dim, lr=1e-3, l2=1e-4, dropout=0.3):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    agent = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout * 0.5),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    agent.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return agent

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    rng = np.random.RandomState(random_state)
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    classes = np.unique(y_arr)
    train_idx = []
    test_idx = []
    for c in classes:
        idx = np.where(y_arr == c)[0].tolist()
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        # ensure at least one sample in train if possible
        if n_test == 0 and len(idx) > 1:
            n_test = 1
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]

def standard_scale_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

def ai_testing(agent, x_test, y_test):
    # ensure numpy arrays and flattened shapes
    x_test_arr = np.array(x_test)
    y_test_arr = np.array(y_test).flatten()

    # evaluate with Keras metrics
    results = agent.evaluate(x_test_arr, y_test_arr, verbose=0)
    print(f'Evaluate results: {results}')

    # predicted probabilities for the positive class
    probs = agent.predict(x_test_arr, verbose=0).flatten()
    preds = (probs >= 0.6).astype(int)
    accuracy = (preds == y_test_arr).mean()
    print(f'Accuracy (threshold 0.6): {accuracy * 100:.2f}%')

    # Plot predicted probabilities and actual labels
    plt.figure(figsize=(10, 4))
    plt.figure(facecolor='#0f172a', edgecolor='#0f172a')
    plt.plot(probs, marker='o', linestyle='-', label='Predicted migraine probability', alpha=0.8)
    plt.scatter(range(len(y_test_arr)), y_test_arr, color='red', marker='x', label='Actual label')
    plt.axhline(0.6, color='gray', linestyle='--', label='Threshold 0.6')
    plt.title(f'Prediction vs Actual — Accuracy {accuracy * 100:.2f}%', color='white')
    plt.xlabel('Sample index', color='white')
    plt.ylabel('Predicted migraine / Actual migraine', color='white')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv('phone_data', names=['datetime', 'steps', 'screen_time', 'stress', 'migraine'],
                     parse_dates=['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour

    X = df[['year', 'month', 'day', 'hour', 'steps', 'screen_time', 'stress']].astype(float)
    y = df['migraine'].astype(int)

    x_train, x_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=42)

    x_train_scaled, x_test_scaled = standard_scale_train_test(x_train, x_test)

    agent = build_binary_classifier(input_dim=x_train_scaled.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
    ]
    agent.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test),
              epochs=200, batch_size=32, callbacks=callbacks, verbose=1)

    agent.save('AI_ai_agent.keras')
    ai_testing(agent, x_test_scaled, y_test)

if __name__ == '__main__':
    main()
