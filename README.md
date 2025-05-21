This paper implements a Generative Adversarial Network (GAN) AI-Driven-Fetal-Distress-Monitoring-SDN-IoMT-Networks.

# Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

Ensure you have:
- Python 3.8+
- Access to a running ONOS controller with REST API enabled

# Configuration

Edit the following line in the Python script to reflect your ONOS IP:

```python
ONOS_CONTROLLER_IP = "127.0.0.1"  # Change to your ONOS IP
```

The default ONOS credentials used:

```
Username: onos
Password: rocks
```

# Model Summary

- **Generator**: Produces synthetic flow samples from random noise.
- **Discriminator**: Learns to distinguish between real and generated samples.
- Trained using binary cross-entropy loss.

# Usage

1. Start ONOS and ensure the REST API is reachable.
2. Run the script:
   ```bash
   python main.py
   ```
3. The model will:
   - Collect flow stats
   - Normalize and train the GAN
   - Detect anomalies
   - Output suspicious flows if detected

# Output

The script prints alerts such as:

```
⚠Anomalous Flows Detected at indices: [2 5 9]
Scores: [0.321, 0.412, 0.298]
```

## Files

- `anomaly_detection_onos.py`: Main detection script.
- `requirements.txt`: Python package requirements.
- `ctg_maternal_dataset.xlsx`: (If integrated with CTG dataset tasks).

##  Tips

- Extend to save alerts to a database or send notifications.
- Tune the threshold based on your network's behavior.
- Add periodic checks using a scheduler (e.g., cronjob or `schedule` module).

---

