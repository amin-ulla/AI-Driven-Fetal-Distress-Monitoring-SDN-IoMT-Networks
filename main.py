# GAN-AE Framework for CTG Anomaly Detection in SDN-IoMT (ONOS-Integrated)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import requests

# === ONOS API Configuration ===
ONOS_API = "http://<ONOS_IP>:8181/onos/v1/flows"  # Replace <ONOS_IP> with actual IP
AUTH = ('onos', 'rocks')
LATENT_DIM = 50

# 1. Load and preprocess the dataset

def load_dataset(path):
    df = pd.read_excel(path)
    X = df.drop(columns=['label'])
    y = df['label'].map({"Normal": 0, "Suspicious": 1, "Pathological": 2})
    return X.values, y.values

def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) + 1e-6)

def normalize_live_data(X_live, X_train):
    return (X_live - X_train.min(axis=0)) / (X_train.max(axis=0) + 1e-6)

# 2. ONOS Flow Fetching

def get_network_data_from_onos():
    try:
        response = requests.get(ONOS_API, auth=AUTH)
        response.raise_for_status()
        flows = response.json().get('flows', [])
        features = []
        for flow in flows:
            pkt_count = flow.get('packetCount', 0)
            byte_count = flow.get('byteCount', 0)
            duration = flow.get('life', 0)
            features.append([pkt_count, byte_count, duration])
        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Error fetching ONOS data: {e}")
        return None

# 3. GAN Models

def build_generator(output_dim):
    return models.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(80, activation='relu'),
        layers.Dense(output_dim, activation='tanh')
    ])

def build_discriminator(input_dim):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(80, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

# 4. Autoencoder

def build_autoencoder(input_dim):
    encoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu')
    ])
    decoder = models.Sequential([
        layers.Input(shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    autoencoder = models.Sequential([encoder, decoder])
    return autoencoder, encoder

# 5. Classifier

def build_classifier(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    return model

# 6. Training Functions

def train_gan(generator, discriminator, real_data, epochs=1000, batch_size=64):
    g_opt = optimizers.Adam(0.0002)
    d_opt = optimizers.Adam(0.0002)
    bce = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(epochs):
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_batch = real_data[idx]
        z = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_batch = generator(z)

        with tf.GradientTape() as tape_d:
            real_out = discriminator(real_batch)
            fake_out = discriminator(fake_batch)
            d_loss = bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)
        grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        with tf.GradientTape() as tape_g:
            fake_batch = generator(z)
            fake_out = discriminator(fake_batch)
            g_loss = bce(tf.ones_like(fake_out), fake_out)
        grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads_g, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss = {d_loss.numpy():.4f}, G Loss = {g_loss.numpy():.4f}")

def train_autoencoder(autoencoder, data, epochs=300):
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=epochs, batch_size=64, verbose=0)

# 7. Anomaly Detection on ONOS

def detect_anomalies_on_onos(classifier, encoder, X_train):
    X_live = get_network_data_from_onos()
    if X_live is not None and len(X_live) > 0:
        X_live_norm = normalize_live_data(X_live, X_train)
        X_encoded = encoder.predict(X_live_norm)
        preds = classifier.predict(X_encoded)
        anomalies = np.where(np.argmax(preds, axis=1) != 0)[0]
        if len(anomalies) > 0:
            print(f"Anomalies detected in ONOS flows at indices: {anomalies}")
        else:
            print("No anomalies detected in ONOS flows.")

# 8. Main Execution

def main():
    X, y = load_dataset("Dataset.xlsx")
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    gan_generator = build_generator(X.shape[1])
    gan_discriminator = build_discriminator(X.shape[1])
    train_gan(gan_generator, gan_discriminator, X_train)

    z = np.random.normal(0, 1, (10000, LATENT_DIM))
    synthetic_data = gan_generator.predict(z)
    X_aug = np.vstack([X_train, synthetic_data])
    y_aug = np.hstack([y_train, np.random.choice([1, 2], 10000)])

    ae, encoder = build_autoencoder(X.shape[1])
    train_autoencoder(ae, X_aug)
    X_encoded = encoder.predict(X_aug)

    classifier = build_classifier(X_encoded.shape[1])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(X_encoded, y_aug, epochs=50, batch_size=64)

    X_test_encoded = encoder.predict(X_test)
    y_pred = np.argmax(classifier.predict(X_test_encoded), axis=1)
    print(classification_report(y_test, y_pred))

    # ONOS Integration: Run anomaly detection on live SDN data
    detect_anomalies_on_onos(classifier, encoder, X_train)

if __name__ == "__main__":
    main()
