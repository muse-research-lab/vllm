import numpy as np
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "results"))
OPT_DATASETS_PATH = os.path.join(ROOT_DIR, "datasets", "opt")

BLOCK_SIZE_TO_COLOR = {
    "1": "#d0e2ff",
    "2": "#a6c8ff",
    "4": "#78a9ff",
    "8": "#4589ff",
    "16": "#0f62fe",
    "32": "#0043ce",
    "64": "#002d9c",
    "128": "#001d6c",
    "256": "#001141"
}

DATASETS_META = {
    "alpaca": {
        "path": os.path.join(OPT_DATASETS_PATH, "alpaca_data.pkl"),
        "color": "#F8DE4B",
        "name": "Alpaca",
    },
    "sharegpt": {
        "path": os.path.join(OPT_DATASETS_PATH, "sharegpt_data.pkl"),
        "color": "#577FBC",
        "name": "ShareGPT",
    },
    "cnn_dailymail": {
        "path": os.path.join(OPT_DATASETS_PATH, "cnn_dailymail_data.pkl"),
        "color": "#E16F65",
        "name": "CNN DailyMail",
    },
    "dolly": {
        "path": os.path.join(OPT_DATASETS_PATH, "dolly_data.pkl"),
        "color": "#57B593",
        "name": "Dolly",
    },
}

MODELS_META = {
    "opt-125m": {
        "name": "OPT-125M"
    },
    "opt-350m": {
        "name": "OPT-350M"
    },
    "opt-1.3b": {
        "name": "OPT-1.3B"
    },
    "opt-2.7b": {
        "name": "OPT-2.7B"
    },
    "opt-6.7b": {
        "name": "OPT-6.7B"
    },
    "opt-13b": {
        "name": "OPT-13B"
    },
    "Llama-2-7b": {
        "name": "Llama-2-7B"
    },
    "Llama-2-13b": {
        "name": "Llama-2-13B"
    },
    "Mistral-7B-v0.1": {
        "name": "Mistral-7B"
    },
    "falcon-7b": {
        "name": "Falcon-7B"
    },
}


def find_median_coord(x, y):
    x_median = np.median(x)
    
    median_idx = 0
    for i, x_val in enumerate(x):
        if x_val >= x_median:
            median_idx = i
            break

    return x_median, y[median_idx]

def find_mean_coord(x, y):
    x_mean = np.mean(x)
    
    mean_idx = 0
    for i, x_val in enumerate(x):
        if x_val >= x_mean:
            mean_idx = i
            break

    return x_mean, y[mean_idx]

def get_cdf(data):
    N = len(data)

    x = np.sort(data)
    y = np.arange(N) / float(N)

    return x, y

def get_tick_positions(x, columns, w, l, L):
    centers = L * np.arange(len(x))
    
    positions = []             
    if (len(columns) % 2) == 0:
        cnt = -1

        for i in range(len(columns)):

            if len(columns) / 2 > i:
                prefix = -1
                cnt += 1
            elif len(columns) / 2 == i:
                prefix = 1
            else:
                prefix = 1
                cnt -= 1

            pos = prefix * ((len(columns) / 2 - cnt) * w + (len(columns) / 2 - cnt - 1) * l + 0.5 * l - 0.5 * w)
                
            positions.append(pos + centers)
    else:
        cnt = -1

        for i in range(len(columns)):

            if (len(columns) / 2 - i) >= 0.5:
                prefix = -1
                cnt += 1
            else:
                prefix = 1
                cnt -= 1

            pos = prefix * ((len(columns) / 2 - cnt) * w + (len(columns) / 2 - 0.5 - cnt) * l - 0.5 * w)

            positions.append(pos + centers)

    return positions