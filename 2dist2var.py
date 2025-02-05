from simulations import simulate
from cmdstanpy import CmdStanModel, write_stan_json
import arviz as az

# Complie stan model
model = CmdStanModel(stan_file="Stan/2dist2var.stan")

# Example usage:
# Ensure p, q, and s are defined
p = 64  # Number of subjects
q = 40  # Number of stimuli
s = 0.5  # Stimulus standard deviation scaling factor

# Run simulation
data = simulate(num_subs=int(p), num_stims=int(q), A_mean=1, B_mean=2, sub_A_sd=1,
            sub_B_sd=1, stim_A_sd=float(s), stim_B_sd=float(s), resid_sd=1,
            ar=[.45, .15], block_size=8)

sub_num = data['sub_num'].values
u0_stim_data = data.iloc[:,:(int(q//2))].sum(axis=1).values
u1_stim_data = data.iloc[:,(int(q)//2):].sum(axis=1).values
u2_stim_data = data.iloc[:, :int(q)//2].values
u3_stim_data = data.iloc[:, (int(q)//2):int(q)].values

stan_data = {
    'N': len(data['y']),
    'n_stims': q//2,
    'n_subs': data['sub_num'].nunique(),
    'sub_num': data['sub_num'].values + 1,
    'u0_stim_data': u0_stim_data,
    'u1_stim_data': u1_stim_data,
    'u2_stim_data': u2_stim_data,
    'u3_stim_data': u3_stim_data,
    'y': data['y'].values,
    'y_t_1': data['y_t-1'].values,
    'y_t_2': data['y_t-2'].values,
    'ar_flag': 1
    }

write_stan_json("data.json", data = stan_data)

fit = model.sample("data.json", chains = 4 , iter_sampling=500, parallel_chains = 4)
# TODO: No warnins given write model check code.
print(az.summary(fit, var_names = ['b', "ar1", "ar2","diff"]))