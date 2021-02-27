import csv
import os
import random
import time
from collections import defaultdict

import lightkurve as lk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow import keras, losses
from tensorflow.keras import layers
from tensorflow.keras.utils import model_to_dot

import transit

# matplotlib.use('TkAgg')

def download_fits_from_csv(csv_file):
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))
        p = tqdm.tqdm(rows)
        hosts = set()
        err = 0
        for row in p:
            host = row['hostname']
            if host in hosts:
                continue
            p.set_description(host)
            try:
                lcf = lk.search_lightcurvefile(host, cadence='short').download()
                lc = lcf.PDCSAP_FLUX.remove_nans().remove_outliers()
                lc.to_fits(f'FITS/{host}.fits', overwrite=True)
            except:
                err += 1
                continue
        print('Encountered', err, 'errors downloading curves')


def extract_training_data(input_samples, model_samples):
    train = input_samples.copy()
    train = np.reshape(train, train.shape + (1,))
    train -= 1
    train /= np.abs(train).max(axis=1)[:,np.newaxis]
    model_y = model_samples.copy()
    model_y = np.reshape(model_y, model_y.shape + (1,))
    model_y -= 1

    norm = np.abs(model_y).max(axis=1)[:,np.newaxis]
    for row in model_y:
        if np.all(row == row[0]):
            row = 0
    model_y /= norm
    model_y = np.nan_to_num(model_y, 0)
    for i in range(model_y.shape[0]):
        if np.all(model_y[i] == -1):
            model_y[i] = 0

    y_true = np.transpose(np.array([train, model_y]), axes=[1, 2, 0, 3])
    return train, y_true

def lightcurve_loss(alpha):
  def loss(true, pred):
    orig = true[:, :, 0, tf.newaxis]
    model = true[:, :, 1, tf.newaxis]
    recon_loss = alpha * tf.reduce_mean(tf.square(orig - pred))
    model_loss = (1 - alpha) * tf.reduce_mean(tf.square(model - pred))
    return  recon_loss + model_loss
  return loss

WINDOW_SIZE = 512

cnn_decoder = keras.Sequential(
    [

    layers.Conv1DTranspose(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(1, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(1, 3, 1, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.Conv1DTranspose(1, 3, 1, activation=layers.LeakyReLU(alpha=0.3), padding='same'),

    layers.Conv1DTranspose(1, 3, 1, activation=layers.LeakyReLU(alpha=0.3), padding='same')

    ]
)
cnn_encoder = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, 1)),
    layers.Dropout(0.5),
    layers.Conv1D(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.MaxPool1D(),
    layers.Conv1D(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),

    layers.MaxPool1D(),
    layers.Conv1D(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(10, 3, 2, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(10, 3, 1, activation=layers.LeakyReLU(alpha=0.3), padding='same'),

    layers.Conv1D(10, 3, 1, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
    layers.Conv1D(1, 3, 1, activation=layers.LeakyReLU(alpha=0.3), padding='same'),
])
cnn_model = keras.Sequential([cnn_encoder, cnn_decoder])


encoder = keras.Sequential([
  layers.Dropout(0.6),
  layers.GRU(8, activation='tanh', return_sequences=True, recurrent_dropout=0.8),
  layers.GRU(8, activation='tanh', return_sequences=True, recurrent_dropout=0.6),
  layers.GRU(8, activation='tanh', return_sequences=False), # Output of this is the encoded vector of the GRU
])

decoder = keras.Sequential([
  layers.RepeatVector(WINDOW_SIZE),
  layers.GRU(8, activation='tanh', return_sequences=True, recurrent_dropout=0.6),
  layers.GRU(8, activation='tanh', return_sequences=True, recurrent_dropout=0.2),
  layers.TimeDistributed(layers.Dense(1)),
])
rnn_model = keras.Sequential(
[
  layers.Input(shape=(WINDOW_SIZE, 1)),
  encoder,
  decoder
])

def test_lightcurves(model):
    tests = [f'HAT-P-{i}' for i in range(1, 30)]
    results_smooth = []
    results_orig = []
    completed = []
    for test in tests:
        try:
            lc_collection = lk.search_lightcurvefile(test, cadence='short').download()
            lc = lc_collection.get_lightcurve('SAP_FLUX')
            samples_per_day = 1 + 60 * 24 // 2
            lco = lc.flatten(samples_per_day)
            lco.plot()
            orig = np.nan_to_num(lco.flux, nan=1.0)
            flat_lc = np.reshape(orig - 1, (1, -1))
            norm = np.abs(flat_lc).max(axis=1)[:,np.newaxis]
            flat_lc /= norm
            padded = np.pad(flat_lc, [(0, 0), (0, -flat_lc.shape[1] % WINDOW_SIZE)], mode='edge')
            output = model(np.reshape(padded, (-1, 512, 1)))

            curve = (output * norm).numpy().reshape(-1) + 1
            plt.figure()
            plt.plot(lc.time, orig, 'b',
                    lc.time, curve[:orig.shape[0]], 'r')
            plt.xlabel(f'Time (Offset, {lc.time_format})')
            plt.ylabel(f'Relative Flux ({str(lc.flux_unit)})')
            plt.legend(['Original', 'CNN Model'])
            plt.title(f'{test}')
            plt.savefig(f'TESTS/{test}-curves.png')
            plt.close()

            smooth = lk.LightCurve(time=lc.time, flux=curve[:orig.shape[0]], label=lco.label)
            smooth_result = transit.fit_model(smooth)
            original_result = transit.fit_model(lco)
            if np.isnan(smooth_result.period) or np.isnan(original_result.period):
                print(f'nan result: {test}, orig period: {original_result.period}, {original_result.snr}. smooth period: {smooth_result.period}, {smooth_result.snr}')
            
            results_smooth.append(smooth_result)
            results_orig.append(original_result)
            fig = plt.figure()
            plt.figure()
            plt.title(f'{test} (Model): period {smooth_result.period:.3f}, depth: {smooth_result.depth:.3f}, snr: {smooth_result.snr:.3f}')
            plt.plot(smooth_result.model_folded_phase, smooth_result.model_folded_model,color='red')
            plt.scatter(smooth_result.folded_phase, smooth_result.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
            plt.xlabel('Time')
            plt.ylabel(f'Relative Flux ({str(lc.flux_unit)})')
            plt.savefig(f'TESTS/{test}-smooth.png')
            plt.close(fig)
            fig = plt.figure()
            plt.title(f'{test} (Original): period {original_result.period:.3f}, depth: {original_result.depth:.3f}, snr: {original_result.snr:.3f}')
            plt.plot(original_result.model_folded_phase, original_result.model_folded_model,color='red')
            plt.scatter(original_result.folded_phase, original_result.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
            plt.xlabel('Time')
            plt.ylabel(f'Relative Flux ({str(lc.flux_unit)})')
            plt.savefig(f'TESTS/{test}-orig.png')
            plt.close(fig)
            completed.append(test)
        except Exception as e:
            print('DNF: ', test)
            print(e)

    
    plt.figure()
    plt.title('SNR Comparison')
    plt.hist([r.snr for r in results_smooth], 30, alpha=0.5, label='Model Results')
    plt.hist([r.snr for r in results_orig], 30, alpha=0.5, label='Original Results')
    plt.legend(loc='upper right')
    plt.savefig('TESTS/snr.png')

    plt.figure()
    plt.title('CDPP Comparison')
    plt.hist([r.CDPP for r in results_smooth], 30, alpha=0.5, label='Model Results')
    plt.hist([r.CDPP for r in results_orig], 30, alpha=0.5, label='Original Results')
    plt.legend(loc='upper right')
    plt.savefig('TESTS/cddp.png')
    for t, r in zip(completed, results_orig):
        print(f'{t}: {r.period}, {r.chi2_min}, {r.CDPP}')
    for t, r in zip(completed, results_smooth):
        print(f'{t}: {r.period}, {r.chi2_min}, {r.CDPP}')


def classify_latent_space(encoder, input_windows, has_transit):
    has_transit_t = np.array([int(v) for v in has_transit])
    encoded_vectors = np.reshape(cnn_encoder(input_windows).numpy(), (-1, 8))
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf.fit(encoded_vectors[:int(0.8*encoded_vectors.shape[0])], has_transit_t[:int(0.8*encoded_vectors.shape[0])])
    test_x, test_cls = encoded_vectors[int(0.8*encoded_vectors.shape[0]):], has_transit_t[int(0.8*encoded_vectors.shape[0]):]
    acc = clf.score(test_x, test_cls)
    print(acc)


def cluster_analysis(encoder, decoder, windows, k=8):
    i = 0
    encoded_vectors = np.reshape(encoder(input_windows).numpy(), (-1, 8))
    centroids = KMeans(n_clusters=k).fit(encoded_vectors).cluster_centers_
    reconstruct = decoder(np.reshape(centroids, (-1, 8, 1)))
    x = np.arange(reconstruct.shape[1])

    for centroid, r in zip(centroids, reconstruct):
        print(centroid)
        plt.plot(x, r)
        plt.xlabel(f'Time (Offset)')
        plt.ylabel(f'Relative Flux')
        plt.title('Centroid Sample')
        plt.savefig(f'TESTS/rnn-cluster{i + 1}.png')
        plt.close()
        i += 1


if __name__ == "__main__":
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Optimizer
        loss=lightcurve_loss(1), # change alpha to affect the fit.
    )
    # rnn_model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=0.01),  # Optimizer
    #     loss=lightcurve_loss(0),
    # )

    # RNN: models/model-cnn-1603336924.9748862.ckpt
    # CNN: models/model-cnn-1603411408.859992.ckpt
    # CNN Part 2: models/model-cnn-1603581786.2035072.ckpt
    # cnn_model.load_weights('models/model-cnn-1603411408.859992.ckpt')

    times, models = transit.read_curves('MODEL')
    curves = transit.read_fits('FITS')
    keys = sorted(times)
    times = [times[k] for k in keys]
    models = [models[k] for k in keys]
    curves = [curves[k] for k in keys]
    input_samples, model_samples, has_transit = transit.samples_lightcurves(curves, times, models, window_size=WINDOW_SIZE, transit_samples_per_curve=40, non_transit_samples_per_curve=40)
    train, y_true = extract_training_data(input_samples, model_samples)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=f'models/model-cnn-{time.time()}.ckpt',
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

    x_train, y_train, x_test, y_test = train[:int(0.8*train.shape[0])],  y_true[:int(0.8*train.shape[0])], train[int(0.8*train.shape[0]):], y_true[int(0.8*train.shape[0]):]
    cnn_model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=50, callbacks=[cp_callback])
    cnn_model.evaluate(x_test, y_test)
    test_lightcurves(cnn_model)
