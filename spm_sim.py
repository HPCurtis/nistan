from simulations import simulate
from cmdstanpy import CmdStanModel, write_stan_json
import arviz as az
import pandas as pd

# Complie stan model
model = CmdStanModel(stan_file="Stan/spm.stan")

# Example usage:
# Ensure p, q, and s are defined
p = 10  # Number of subjects
q = 40  # Number of stimuli
s = 0.5  # Stimulus standard deviation scaling factor


# Run simulation
data = simulate(num_subs=int(p), num_stims=int(q), A_mean=1, B_mean=2, sub_A_sd=1,
            sub_B_sd=1, stim_A_sd=float(s), stim_B_sd=float(s), resid_sd=1,
            ar=[.45, .15], block_size=8)


# Grouping by 'Department'
grouped = data.groupby('sub_num')

# Creating separate DataFrames for each group
separate_dfs = {dept: group for dept, group in grouped}

def spm_design_matrix(df):
    y = df['y'].values
    X = pd.concat([df.iloc[:,:int(q)//2].sum(axis=1),
                       df.iloc[:,q//2:int(q)].sum(axis=1),
                       df['y_t-1'],
                       df['y_t-2']], axis=1).values
    return(y,X)


diffs = []

for subj, subj_df in separate_dfs.items():
    y, X = spm_design_matrix(subj_df)
   
    stan_data = {
        "N": len(y),
        "K": X.shape[1],
        "y": y,
        "X": X
    }
    
    write_stan_json("spm_data.json", data = stan_data)
    
    fit = model.sample("spm_data.json", chains = 2, iter_sampling=500)
    idata = az.from_cmdstanpy(fit)
    diffs.append(idata.posterior)

print(az.summary(diffs[0]))