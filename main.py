import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
import requests

latent_dim = 128
ONOS_IP = "127.0.0.1"
ONOS_API = f"http://{ONOS_IP}:8181/onos/v1/flows"
AUTH = ('onos', 'rocks')
DATASET_PATH = "Dataset.xlsx"

def build_generator(output_dim):
    return tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(latent_dim,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(output_dim, activation="tanh")
    ])

def build_discriminator(input_dim):
    return tf.keras.Sequential([
        layers.Dense(1024, activation="relu", input_shape=(input_dim,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

def load_dataset(path):
    df = pd.read_excel(path)
    df = df.drop(columns=["label"], errors="ignore")
    return df.values.astype(np.float32)

def normalize(data, ref_min=None, ref_max=None):
    if ref_min is None or ref_max is None:
        ref_min, ref_max = data.min(axis=0), data.max(axis=0)
    return (data - ref_min) / (ref_max + 1e-6), ref_min, ref_max

def get_network_data():
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
        print(f"Failed to fetch ONOS data: {e}")
        return None

def train_began(generator, discriminator, data, epochs=1000, batch_size=64, lr=0.0002):
    g_optimizer = optimizers.Adam(lr)
    d_optimizer = optimizers.Adam(lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_samples = data[idx]
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator(z, training=True)

        with tf.GradientTape() as tape_d:
            real_preds = discriminator(real_samples, training=True)
            fake_preds = discriminator(fake_samples, training=True)
            d_loss = loss_fn(tf.ones_like(real_preds), real_preds) + \
                     loss_fn(tf.zeros_like(fake_preds), fake_preds)
        grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        with tf.GradientTape() as tape_g:
            generated = generator(z, training=True)
            fake_preds = discriminator(generated, training=True)
            g_loss = loss_fn(tf.ones_like(fake_preds), fake_preds)
        grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss.numpy():.4f} - G Loss: {g_loss.numpy():.4f}")

def detect_anomalies(discriminator, data, threshold=0.5):
    scores = discriminator(data).numpy().flatten()
    anomalies = np.where(scores < threshold)[0]
    if len(anomalies) > 0:
        print(f"Anomalous Flows Detected: {anomalies.tolist()}")
    else:
        print("No Anomalous Flows Detected.")

def main():
    dataset = load_dataset(DATASET_PATH)
    if dataset is None or len(dataset) < 10:
        print("Not enough dataset samples.")
        return

    norm_dataset, ref_min, ref_max = normalize(dataset)

    generator = build_generator(norm_dataset.shape[1])
    discriminator = build_discriminator(norm_dataset.shape[1])

    print("Training GAN on CTG dataset...")
    train_began(generator, discriminator, norm_dataset)

    print("Fetching ONOS flow data for inference...")
    network_data = get_network_data()
    if network_data is None or len(network_data) < 3:
        print("Not enough ONOS flow data.")
        return

    norm_net_data, _, _ = normalize(network_data, ref_min, ref_max)
    print("Detecting anomalies in ONOS flow data...")
    detect_anomalies(discriminator, norm_net_data)

if __name__ == "__main__":
    main()
