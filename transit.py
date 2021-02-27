from transitleastsquares import transitleastsquares, resample
from matplotlib import pyplot as plt
import random
import itertools
from fastdtw import fastdtw
import numpy as np
import csv
from collections import namedtuple
import lightkurve as lk
import os


def print_results(results):
  print('Period', format(results.period, '.5f'), 'd')
  print(len(results.transit_times), 'transit times in time series:', \
          ['{0:0.5f}'.format(i) for i in results.transit_times])
  print('Transit depth', format(results.depth, '.5f'))
  print('Transit duration (days)', format(results.duration, '.5f'))
  print('SNR', results.snr)
  print('chi2', results.chi2_min)
  print('CDPP', results.CDPP)

def plot_result_fit(results):
  plt.figure()
  # plt.plot(results.model_lightcurve_time, results.model_lightcurve_model)
  # plt.scatter(smooth.time, smooth.flux)
  plt.plot(results.model_folded_phase, results.model_folded_model,color='red')
  plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
  plt.xlabel('Time')
  plt.ylabel('Relative flux')
  plt.show()

def fit_and_report(lc):
  results = fit_model(lc)
  print_results(results)
  # plot_results(results)
  plot_result_fit(results)

Transit = namedtuple('Transit', ['host', 'period', 'radius', 'mass'])

def read_stats(csv_file):
    stats = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            hostname = row['hostname']
            if hostname in stats:
                continue
            stats[hostname] = Transit(hostname, float(row['pl_orbper']), float(row['pl_radj']), float(row['pl_bmassj'] or '-1'))
    return stats

def fit_model(lc):
  pg = lc.to_periodogram(method='bls',duration=0.2, minimum_period=0.5)

  model = transitleastsquares(lc.time, lc.flux)
  results = model.power(period_min=pg.period_at_max_power.value*0.9, period_max=pg.period_at_max_power.value*1.1)
  results.CDPP = lc.estimate_cdpp()
  return results


def match_transits(lcs, stats):
    verified_lcs = []
    results = {}

    for host, lcf in list(lcs.items()):
        record = stats[host]
        samples_per_day = 1 + 60 * 24 // 2
        lc = lcf.FLUX.flatten(samples_per_day)
        result = fit_model(lc)
        if not np.isnan(result.period) and np.isclose(record.period, result.period, atol=0.5):
            verified_lcs.append(lc)
            results[host] = result
    return verified_lcs, results


def samples_lightcurves(lcs, model_times, models, window_size=512, transit_samples_per_curve=35, non_transit_samples_per_curve=20):
    ''' Return a subsampled lightcurve and result array that can be fed to training. '''
    samples_per_curve = transit_samples_per_curve + non_transit_samples_per_curve
    input_fit = np.zeros((samples_per_curve * len(lcs), window_size))
    model_fit = np.zeros((samples_per_curve * len(lcs), window_size))
    has_transit = []
    i = 0
    for lc, model_time, model in zip(lcs, model_times, models):

        # Down sample the result curve by a factor of five (the amount that the transitleastsquares upsamples by).
        samples_per_day = 1 + 60 * 24 // 2
        lc = lc.FLUX.flatten(samples_per_day)
        matching_model_time = np.searchsorted(model_time, lc.time)[:-1]
        result_flux = model[matching_model_time]
        lc_flux = lc.flux[:-1]
        # select all positions in time_new that match
        # We want to sample around transits so here we ensure that transits are in every sample we pick
        transit_events = np.where(result_flux != 1)[0]
        # We need to only take transit events that can fit a window_size buffer around them
        threshold = min(lc_flux.shape[0], result_flux.shape[0]) - window_size
        # select k=transit_samples_per_curve positions in the set of places where a transit is occuring
        starts = random.choices(transit_events[np.where(transit_events < threshold)[0]], k=transit_samples_per_curve)
        has_transit += [True] * len(starts)
        # select k=non_transit_samples_per_curve positions where a transit *might* be happening (but in all liklihood probably not)
        rand_starts = random.choices(list(range(window_size, lc_flux.shape[0] - window_size)), k=non_transit_samples_per_curve)
        has_transit += [False] * len(starts)

        # for each position pick a window centred about it
        slices = np.array([np.arange(s - window_size // 2, s + window_size // 2) for s in starts + rand_starts])

        input_fit[i:i + samples_per_curve] = lc_flux[slices]
        model_fit[i:i + samples_per_curve] = result_flux[slices]
        i += samples_per_curve
    return input_fit, model_fit, has_transit


def read_fits(dir):
    return {f.rstrip('.fits'): lk.open(f'{dir}/{f}') for f in os.listdir(dir)}

def read_curves(dir):
    models = {f.rstrip('.npy'): np.load(f'{dir}/{f}') for f in os.listdir(dir) if 'time' not in f}
    times = {f.rstrip('-time.npy'): np.load(f'{dir}/{f}') for f in os.listdir(dir) if 'time' in f}
    return times, models