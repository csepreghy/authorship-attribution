import pandas as pd
import time as time

from neural_network_classifier import run_neural_network

df = pd.read_hdf('data/FAKE_train.h5')
model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)
