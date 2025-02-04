from simulations import simulate
from cmdstanpy import CmdStanModel
# Complie stan model
model = CmdStanModel(stan_file=".\Stan\fixed_nc.stan")

# Example usage:
# Ensure p, q, and s are defined
p = 10  # Number of subjects
q = 40  # Number of stimuli
s = 0.5  # Stimulus standard deviation scaling factor


# Run simulation
data = simulate(num_subs=int(p), num_stims=int(q), A_mean=1, B_mean=2, sub_A_sd=1,
            sub_B_sd=1, stim_A_sd=float(s), stim_B_sd=float(s), resid_sd=1,
            ar=[.45, .15], block_size=8)

sub_num = data['sub_num'].values
u0_stim_data = data.iloc[:,:(int(q//2))].sum(axis=1).values
u1_stim_data = data.iloc[:,(int(q)//2):].sum(axis=1).values
stim_data = data.iloc[:, :int(q)]