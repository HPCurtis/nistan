# Import required packages
import numpy as np
import pandas as pd
import scipy.stats as sp
from random import shuffle

# Double gamma HRF function
def spm_hrf(TR, p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution

    Arguments:

    Required:
    TR: repetition time at which to generate the HRF (in seconds)

    Optional:
    p: list with parameters of the two gamma functions:
                                                 defaults
                                                (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32
    """

    p = [float(x) for x in p]

    fMRI_T = 16.0

    TR = float(TR)
    dt  = TR / fMRI_T
    u   = np.arange(p[6] / dt + 1) - p[5] / dt
    hrf = sp.gamma.pdf(u, p[0] / p[2], scale=1.0 / (dt / p[2])) - sp.gamma.pdf(u, p[1] / p[3], scale=1.0 / (dt / p[3])) / p[4]
    good_pts = np.array(range(int(p[6] / TR))) * fMRI_T
    good_pts = np.floor(good_pts).astype(int)  # Convert to integers
    hrf = hrf[good_pts]  # Now it will be valid indexing
    hrf = hrf / np.sum(hrf)
    return hrf

# Function to insert ISIs into a trial list
def insert_ISI(trials, ISI):
    return np.insert(trials, np.repeat(range(1, len(trials)), ISI), 0)

# Function to build activation sequence from stimulus list
def build_seq(sub_num, stims, sub_A_sd, sub_B_sd):
    stims = stims.reindex(np.random.permutation(stims.index))

    # Randomize ISI between 1, 2, 3, and 4, removing the first one
    ISI = np.delete(np.repeat([1, 2, 3, 4], len(stims.index) // 4, axis=0), 0)
    np.random.shuffle(ISI)

    # Create matrix of stimulus predictors and add ISIs
    X = np.diag(stims['effect'])
    X = np.apply_along_axis(func1d=insert_ISI, axis=0, arr=X, ISI=ISI)

    # Reorder the columns so they are in the same order (0-39) for everyone
    X = X[:, [list(stims['stim']).index([i]) for i in range(len(stims.index))]]

    # Convolve all predictors with the double gamma HRF
    X = np.apply_along_axis(func1d=np.convolve, axis=0, arr=X, v=spm_hrf(1))

    # Build and return this subject's dataframe
    df = pd.DataFrame(X)
    df['time'] = range(len(df.index))
    df['sub_num'] = sub_num
    df['sub_A'] = np.random.normal(size=1, scale=sub_A_sd)
    df['sub_B'] = np.random.normal(size=1, scale=sub_B_sd)
    return df

# Function to build activation sequence for block design
def build_seq_block(sub_num, stims, sub_A_sd, sub_B_sd, block_size):
    q = len(stims.index)
    stims = [stims.iloc[:q // 2,], stims.iloc[q // 2:,]]
    stims = [x.reindex(np.random.permutation(x.index)) for x in stims]
    shuffle(stims)
    stims = [[x.iloc[k:(k + block_size),] for k in range(0, q // 2, block_size)] for x in stims]
    stims = pd.concat([val for pair in zip(stims[0], stims[1]) for val in pair])

    # Randomize ISI between 1, 2, 3, and 4, removing the first one
    ISI = np.delete(np.repeat(2, len(stims.index), axis=0), 0)

    # Create matrix of stimulus predictors and add ISIs
    X = np.diag(stims['effect'])
    X = np.apply_along_axis(func1d=insert_ISI, axis=0, arr=X, ISI=ISI)

    # Reorder the columns
    X = X[:, [list(stims['stim']).index(i) for i in range(len(stims.index))]]

    # Convolve all predictors with the double gamma HRF
    X = np.apply_along_axis(func1d=np.convolve, axis=0, arr=X, v=spm_hrf(1))

    # Build and return this subject's dataframe
    df = pd.DataFrame(X)
    df['time'] = range(len(df.index))
    df['sub_num'] = sub_num
    df['sub_A'] = np.random.normal(size=len(df), scale=sub_A_sd)
    df['sub_B'] = np.random.normal(size=len(df), scale=sub_B_sd)
    return df

# Generalize the code into a simulation function
def simulate(num_subs, num_stims, A_mean, B_mean, sub_A_sd, sub_B_sd, stim_A_sd,
             stim_B_sd, resid_sd, ar=None, block_size=None):
    # Build stimulus list
    stims = np.random.normal(size=num_stims // 2, loc=1, scale=stim_A_sd / A_mean).tolist() + \
            np.random.normal(size=num_stims // 2, loc=1, scale=stim_B_sd / B_mean).tolist()
    stims = pd.DataFrame({'stim': range(num_stims),
                          'condition': np.repeat([0, 1], num_stims // 2),
                          'effect': np.array(stims)})

    # Build design matrix from stimulus list
    if block_size is None:
        # Build event-related design
        data = pd.concat([build_seq(sub_num=i, stims=stims, sub_A_sd=sub_A_sd, sub_B_sd=sub_B_sd) for i in range(num_subs)])
    else:
        # Build blocked design
        data = pd.concat([build_seq_block(sub_num=i, stims=stims, sub_A_sd=sub_A_sd, sub_B_sd=sub_B_sd, block_size=block_size) for i in range(num_subs)])

    # Add response variable and difference predictor
    if ar is None:
        # Build y WITHOUT AR(2) errors
        data['y'] = (A_mean + data['sub_A']) * data.iloc[:, :(num_stims // 2)].sum(axis=1).values + \
                    (B_mean + data['sub_B']) * data.iloc[:, (num_stims // 2):num_stims].sum(axis=1).values + \
                    np.random.normal(size=len(data.index), scale=resid_sd)
    else:
        # Build y WITH AR(2) errors
        data['y'] = np.empty(len(data.index))
        data['y_t-1'] = np.zeros(len(data.index))
        data['y_t-2'] = np.zeros(len(data.index))
        for t in range(len(pd.unique(data['time']))):
            data.loc[t, 'y'] = pd.DataFrame(
                (A_mean + data.loc[t, 'sub_A']) * data.loc[t, range(num_stims // 2)].sum(axis=1).values + \
                (B_mean + data.loc[t, 'sub_B']) * data.loc[t, range(num_stims // 2, num_stims)].sum(axis=1).values + \
                np.random.normal(size=len(data.loc[t].index), scale=resid_sd)).values
            if t == 1:
                data.loc[t, 'y'] = pd.DataFrame(data.loc[t, 'y'].values + ar[0] * data.loc[t - 1, 'y'].values).values
                data.loc[t, 'y_t-1'] = pd.DataFrame(data.loc[t - 1, 'y']).values
            if t > 1:
                data.loc[t, 'y'] = pd.DataFrame(data.loc[t, 'y'].values + ar[0] * data.loc[t - 1, 'y'].values + ar[1] * data.loc[t - 2, 'y'].values).values
                data.loc[t, 'y_t-1'] = pd.DataFrame(data.loc[t - 1, 'y']).values
                data.loc[t, 'y_t-2'] = pd.DataFrame(data.loc[t - 2, 'y']).values

    # Remove random stimulus effects from regressors before fitting model
    data.iloc[:, :num_stims] = data.iloc[:, :num_stims] / stims['effect'].tolist()

    # Build design DataFrame (when stimulus was presented for each subject)
    gb = data.groupby('sub_num')
    pres = pd.DataFrame([[next(i - 1 for i, val in enumerate(df.iloc[:, stim]) if abs(val) > .0001)
                          for stim in range(num_stims)] for sub_num, df in gb])

    # Build the design DataFrame from pres
    design = pd.concat([pd.DataFrame({'onset': pres.iloc[sub, :].sort_values(),
                                      'run_onset': pres.iloc[sub, :].sort_values(),
                                      'stimulus': pres.iloc[sub, :].sort_values().index,
                                      'subject': sub,
                                      'duration': 1,
                                      'amplitude': 1,
                                      'run': 1,
                                      'index': range(pres.shape[1])})
                        for sub in range(num_subs)])
    design['condition'] = stims['condition'][design['stimulus']]

    # Build activation DataFrame
    activation = pd.DataFrame({'y': data['y'].values,
                               'vol': data['time'],
                               'run': 1,
                               'subject': data['sub_num'] + 1})

    return data


if __name__ == "__main__":
    # Example usage:
    # Ensure p, q, and s are defined
    p = 10  # Number of subjects
    q = 40  # Number of stimuli
    s = 0.5  # Stimulus standard deviation scaling factor


    # Run simulation
    dat = simulate(num_subs=int(p), num_stims=int(q), A_mean=1, B_mean=2, sub_A_sd=1,
                sub_B_sd=1, stim_A_sd=float(s), stim_B_sd=float(s), resid_sd=1,
                ar=[.45, .15], block_size=8)

    print(dat)