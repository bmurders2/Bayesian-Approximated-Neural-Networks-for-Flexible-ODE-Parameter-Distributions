'''
Author: 
    Benjamin Murders
Inspired Paper: 
    https://arxiv.org/pdf/1506.02142v4.pdf 
    (Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning)

Purpose: 
    Working proof of concept for applying Bayesian approximated neural networks for estimating the distribution of function parameters. 
    In this case, approximating two parameters for an ODE in the form of: dy/dt = -r * y.

Required dependencies:
    numpy
    pandas
    pickle
    matplotlib
    seaborn
    jax[cpu] (JAX)
        *GPU with CUDA package is preferable but not necessary
        I used the Docker image [tensorflow/tensorflow:latest-gpu] as the base Linux image via WSL 
            Python 3.6.9
            Tensorflow Version: 2.6.0
            https://www.tensorflow.org/install/gpu
            https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute


Note: 
    -The character string, #%%, is a feature in VS Code for defining Jupyter-like cells within the Interactive Window.
        https://code.visualstudio.com/docs/python/jupyter-support-py
    -This script also includes archiving the sample data out to CSV files as well as the trained weights for re-sampling
        without continued training if needed. 

'''
#%%
# imports
import os
import numpy as np
import pandas as pd
import pickle

from jax.experimental import ode as jax_ode_solver
from jax import numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Conv, Dense, Dropout, MaxPool, Relu, Flatten, LogSoftmax, elementwise
from jax.interpreters.batching import batch

import seaborn as sns
from matplotlib import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

from jax.config import config


#%%
# functions
# activation functions
@jit
def softplus(z):
    return jnp.log(1. + jnp.exp(z))
@jit
def mish(z):
    return z * jnp.tanh(softplus(z))
@jit
def elu(z, alpha: float = 1., offset: float = 0.):
    # when "offset" parameter is >= 1,
    #   R>0 = { x∈R ∣ x>0 }
    return jnp.where(z < 0., alpha * (jnp.exp(z) - 1. + offset), z + offset)

# main differential equation
@jit
def diff_eq(y, t, params):
    alpha = -1. * jnp.abs(params)
    dy = alpha * y
    return dy

def generate_ode_func_data(Y0, t_eval, ode_func_params, ode_func):
    ode_results = jax_ode_solver.odeint(ode_func, Y0, t_eval, ode_func_params).ravel()
    return ode_results

def generate_data_batch(initial_Y0, t_eval, ode_func_params, ode_func):
    gen_data_params = {'Y0': initial_Y0, 't_eval': t_eval, 'ode_func_params': ode_func_params, 'ode_func': ode_func}
    return (t_eval, generate_ode_func_data(**gen_data_params))

def slice_actual_data_batch(data_batch, size, add_noise_func = None, add_noise_params = None):
    def noise(input, params, noise_func): return noise_func(input, **params) if params != None else noise_func(input)
    def sort_by_t(t_, y_): 
        input_array = jnp.array([t_, y_])
        sorted_indices_by_t = input_array[0,:].argsort()
        sorted_t_, sorted_y_ = [input_array[input_arg,sorted_indices_by_t] for input_arg in range(2)]
        return sorted_t_, sorted_y_

    t_slices = jnp.sort(np.random.choice(data_batch[0], size=size, replace=False))
    y_slices = data_batch[1][t_slices.astype(jnp.int64)]
    if add_noise_func != None:
        t_slices, y_slices = [noise(slices_, add_noise_params, noise_func=add_noise) for slices_ in [t_slices, y_slices]]

    # make sure t steps are in order for ode solver and keep matching index for y
    t_slices, y_slices = sort_by_t(t_slices, y_slices)
    # return slices and make sure y > 0 given
    return (t_slices, elu(y_slices, offset=1.))

def add_noise(input, offset: float = 1.25, scale: float = 1.25):
    def random_noise(mean, noise_scale):
        return np.random.normal(mean, noise_scale)

    delta_t = offset + scale * np.random.random(input.shape)
    input += random_noise(mean=0.0, noise_scale=delta_t)
    return input


# huber loss
@jit
def huber(target, pred, delta=1e-1):
    loss = jnp.where(jnp.abs(target-pred) < delta , 0.5*((target-pred)**2), delta*jnp.abs(target - pred) - 0.5*(delta**2))
    return jnp.mean(loss)
@jit
def mse(target, pred, scale = 0.5 ):
    loss = scale * jnp.power(pred - target, 2)
    return jnp.mean(loss)

# priming functions
@jit
def primer_loss(params, batch, key, nn_inputs, loss_func=huber):
    inputs, targets = batch
    nn_ode_func_params = apply_fun(params, nn_inputs, rng=key)[0,:]

    log_targets, log_nn_ode_func_params = [elu(jnp.log(_),offset=1.) for _ in [targets, nn_ode_func_params]]
    loss_result = loss_func(log_targets, log_nn_ode_func_params)
    return loss_result

@jit
def primer_step(i, opt_state, batch, key, nn_inputs):
    params = get_params(opt_state)
    g = grad(primer_loss)(params, batch, key, nn_inputs)
    return opt_update(i, g, opt_state)

def prime_params(initial_guess, batch, net_params, train_iter, nn_inputs, key):
    print('\nPriming params for nn output for initial guess of differential equation input params')
    opt_state = opt_init(net_params)
    for i in range(train_iter):
        key, subkey = random.split(key)
        opt_state = primer_step(i, opt_state, batch, subkey, nn_inputs)

        if i % int(max(train_iter, 10.) /10.) == 0:
            step_loss = primer_loss(get_params(opt_state), batch, key, nn_inputs)
            print('loss at step {step_i}: {value}'.format(step_i=i,value=step_loss))

    final_loss_value = primer_loss(get_params(opt_state), batch, key, nn_inputs)
    print('\nInitial guess loss at step {step_i}: {value}'.format(step_i=i, value=final_loss_value))
    net_params = get_params(opt_state)
    print_tmp_values = apply_fun(net_params, nn_inputs, rng=key)[0,:]
    print('Initial guess values (trained): {values}'.format(values=jnp.round(print_tmp_values,4)))
    print('Initial guess values (target): {values}'.format(values=jnp.round(initial_guess,4)))

    return net_params, key

# training functions
@jit
def train_loss(params, batch, key, nn_inputs, loss_func = huber):
    t, targets = batch

    nn_ode_func_params = apply_fun(params, nn_inputs, rng=key)[0,:]
    Y0, ode_func_params = nn_ode_func_params

    ode_results = jax_ode_solver.odeint(diff_eq, Y0, t, ode_func_params, rtol=2e-7, atol=2e-7, mxstep=500)
    log_targets, log_ode_results = [elu(jnp.log(_),offset=1.) for _ in [targets, ode_results]]
    loss_result = loss_func(log_targets, log_ode_results)
    return loss_result

@jit
def train_step(i, opt_state, batch, key, nn_inputs):
  params = get_params(opt_state)
  g = grad(train_loss)(params, batch, key, nn_inputs)
  return opt_update(i, g, opt_state)

def train_params(batch, net_params, opt_state, train_iter, nn_inputs, key):
    print('\nTraining Model')
    for i in range(train_iter):
        key, subkey = random.split(key)
        opt_state = train_step(i, opt_state, batch, subkey, nn_inputs)

        if i % int(max(train_iter, 10.) /10.) == 0:
            step_loss = train_loss(get_params(opt_state), batch, key, nn_inputs)
            print('loss at step {step_i}: {value}'.format(step_i=i,value=step_loss))

    # print training details
    print('\nTraining Complete')
    final_loss_value = train_loss(get_params(opt_state), training_data_batch, key, nn_inputs)
    print('Training loss at step {step_i}: {value}'.format(step_i=train_iter,value=final_loss_value))
    net_params = get_params(opt_state)

    return net_params, key

def sample_model(opt_state, sample_count, t_eval, nn_inputs, key):
    print('\nSampling Model Output Distribution and Param(s) Distribution(s) | Sample Count: {sample_cnt}'.format(sample_cnt=sample_count))
    pred_list = list()
    theta_list = list()
    net_params = get_params(opt_state)
    for sample in range(sample_count):
        key, subkey = random.split(key)
        nn_ode_func_params = apply_fun(net_params, nn_inputs, rng=key)[0,:]
        theta_list.append(nn_ode_func_params)

        Y0, ode_func_params = nn_ode_func_params
        ode_results = jax_ode_solver.odeint(diff_eq, Y0, t_eval, ode_func_params)
        pred_list.append(ode_results)

    thetas = jnp.array(theta_list)
    return thetas, theta_list, pred_list

def convert_samples_to_dataframe(thetas, pred_list, param_count, sample_count, file_path):
    print('\nOrganizing/structuring model output and sample data to Pandas dataframes')
    pred_list_data = jnp.array(pred_list)

    theta_df = pd.DataFrame(data=thetas, columns=["theta_{value}".format(value=_) for _ in range(param_count)])
    pred_df = pd.DataFrame(data=pred_list_data.T, columns=["sample_{value}".format(value=_) for _ in range(sample_count)])
    pred_df['index'] = pred_df.index
    pred_df_density = pd.melt(pred_df, id_vars=['index'])

    pred_sample_col_list = [col_name for col_name in pred_df.columns.tolist() if 'sample_' in col_name]
    theta_col_list = [col_name for col_name in theta_df.columns.tolist() if 'theta_' in col_name]
    sample_count = len(pred_sample_col_list)
    param_count = len(theta_col_list)

    pred_df['mean'] = pred_df[pred_sample_col_list].mean(axis=1)
    pred_df['std_dev'] = pred_df[pred_sample_col_list].std(axis=1)
    pred_df['quantile_025'] = pred_df[pred_sample_col_list].dropna().quantile(0.025,axis=1)
    pred_df['quantile_050'] = pred_df[pred_sample_col_list].dropna().quantile(0.05,axis=1)
    pred_df['quantile_250'] = pred_df[pred_sample_col_list].dropna().quantile(0.25,axis=1)
    pred_df['quantile_500'] = pred_df[pred_sample_col_list].dropna().quantile(0.50,axis=1)
    pred_df['quantile_750'] = pred_df[pred_sample_col_list].dropna().quantile(0.75,axis=1)
    pred_df['quantile_950'] = pred_df[pred_sample_col_list].dropna().quantile(0.95,axis=1)
    pred_df['quantile_975'] = pred_df[pred_sample_col_list].dropna().quantile(0.975,axis=1)

    [dff.to_csv(os.path.join(py_file_path,'{file_name}.csv'.format(file_name=export_file_name))) for dff, export_file_name in 
        [
            [pred_df,'pred_df'],
            [theta_df,'theta_df'],
            [pred_df_density,'pred_df_density']
        ]
    ]
    return theta_df, pred_df, pred_df_density, pred_sample_col_list



# plotting functions
def main_density_plot(t_eval, sample_count, pred_df, actual_data_df, train_data_df, pred_df_density):
    mpl.rcParams['figure.figsize'] = [24,12]
    mpl.rcParams['figure.dpi'] = 240
    fig, ax = plt.subplots()
    x_min = 0.
    x_max = max(t_eval)
    y_min = -1.
    y_max = max(actual_data_df['y'].to_numpy().max()*1.1,pred_df_density['value'].to_numpy().max()*1.1)

    # mean
    ax.plot(t_eval, pred_df['mean'], linestyle='solid',color='white',alpha=1.,lw=3.,zorder=0,label='_mean')
    ax.plot(t_eval, pred_df['mean'], linestyle='solid',color='brown',alpha=1,lw=2.,zorder=1,label='Mean')

    # median
    ax.plot(t_eval, pred_df['quantile_500'], linestyle='solid',color='white',alpha=1.,lw=4.,zorder=2,label='_median')
    ax.plot(t_eval, pred_df['quantile_500'], linestyle='solid',color='green',alpha=1,lw=3.,zorder=3,label='Median')

    # actual data
    ax.scatter(actual_data_df['t'], actual_data_df['y'], label='_actual_data',color='white',lw=5.,zorder=4)
    ax.scatter(actual_data_df['t'], actual_data_df['y'], label='Actual Data',color='coral',lw=3.5,zorder=5)

    # train data
    ax.scatter(train_data_df['t'], train_data_df['y'], label='_traing_data',color='white',lw=5.,zorder=6)
    ax.scatter(train_data_df['t'], train_data_df['y'], label='Train Data',color='black',lw=3.5,zorder=7)


    sns.kdeplot(data=pred_df_density, x='index', y='value', fill=True, thresh=0.0, levels=100, cmap="mako", cbar=True)

    plt.title('Density Plot - [dy/dt = -r * y] | {sample_cnt} Samples'.format(sample_cnt=sample_count))
    plt.xlabel('t')
    plt.ylabel('f(t)')
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min,x_max)
    plt.legend()
    ax.grid()
    plt.show()

# Plot theta distributions
def plot_theta_density(theta_df,col,xlabel:str=None):
    mpl.rc_file_defaults()
    mpl.rcParams['figure.figsize'] = [24,12]
    mpl.rcParams['figure.dpi'] = 240
    sns.set_theme()
    title_str = 'ODE Param KDE'
    title_str = title_str if not xlabel else '{title_string} - Theta Param: {theta_param_name}'.format(title_string=title_str,theta_param_name=xlabel)

    df_col_name = "theta_{col_ref}".format(col_ref=col)
    dff_theta = theta_df[df_col_name]
    
    data_np = dff_theta.to_numpy()
    clip_range_min = 0
    clip_range_max = 100*data_np.std(axis=0) + data_np.mean(axis=0)

    sns.kdeplot(data=dff_theta, fill=True, clip=(clip_range_min,clip_range_max))
    sns.rugplot(data=dff_theta,color='r')    
    # sns.histplot(dff_theta,color='g')
    # plt.axvline(dff_theta.mode(),label='Mode: {val}'.format(val=jnp.round(dff_theta.mode(),4)),color='black')

    plt.axvline(dff_theta.dropna().quantile(0.50),label='Median: {val}'.format(val=jnp.round(dff_theta.dropna().quantile(0.50),4)),color='green')
    plt.axvline(data_np.mean(),label='Mean: {val}'.format(val=jnp.round(data_np.mean(),4),color='blue'))

    plt.axvline(dff_theta.dropna().quantile(0.025),label='2.5th Percentile: {val}'.format(val=jnp.round(dff_theta.dropna().quantile(0.025),4)),color='black')
    plt.axvline(dff_theta.dropna().quantile(0.25),label='25th Percentile: {val}'.format(val=jnp.round(dff_theta.dropna().quantile(0.25),4)),color='purple')
    plt.axvline(dff_theta.dropna().quantile(0.75),label='75th Percentile: {val}'.format(val=jnp.round(dff_theta.dropna().quantile(0.75),4)),color='purple')
    plt.axvline(dff_theta.dropna().quantile(0.975),label='97.5th Percentile: {val}'.format(val=jnp.round(dff_theta.dropna().quantile(0.975),4)),color='black')


    frame = pylab.gca()
    frame.axes.get_yaxis().set_ticks([])
    plt.legend()
    plt.title(title_str)
    plt.ylabel('')
    plt.xlabel(xlabel if xlabel else df_col_name)
    plt.xticks(rotation=45)
    plt.show()

# plot - density contour plot
def plot_contour(t_eval, actual_data_df, pred_df, pred_df_density, sample_count):
    mpl.rcParams['figure.figsize'] = [24,12]
    mpl.rcParams['figure.dpi'] = 240
    fig, ax = plt.subplots()
    x_min = 0.
    x_max = max(t_eval)
    y_min = -1.
    y_max = max(actual_data_df['y'].to_numpy().max()*1.1,pred_df_density['value'].to_numpy().max()*1.1)

    # median
    ax.plot(t_eval, pred_df['quantile_500'], linestyle='solid',color='white',alpha=1.,lw=4.,zorder=2,label='_median')
    ax.plot(t_eval, pred_df['quantile_500'], linestyle='solid',color='green',alpha=1,lw=3.,zorder=3,label='Median')

    # actual data
    ax.scatter(actual_data_df['t'], actual_data_df['y'], label='_actual_data',color='white',lw=5.,zorder=4)
    ax.scatter(actual_data_df['t'], actual_data_df['y'], label='Actual Data',color='coral',lw=3.5,zorder=5)

    # train data
    ax.scatter(train_data_df['t'], train_data_df['y'], label='_traing_data',color='white',lw=5.,zorder=6)
    ax.scatter(train_data_df['t'], train_data_df['y'], label='Train Data',color='black',lw=3.5,zorder=7)

    # density - contour
    sns.kdeplot(data=pred_df_density, x='index', y='value',cbar=True,fill=True,color='lightblue',zorder=0)

    plt.title('Density Plot - Contour View | {sample_cnt} Samples'.format(sample_cnt=sample_count))
    plt.xlabel('t')
    plt.ylabel('f(t)')
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min,x_max)
    ax.grid()
    plt.legend()
    plt.show()

# plot - individual samples view
def plot_samples(t_eval, actual_data_df, pred_df, pred_sample_col_list):
    mpl.rcParams['figure.figsize'] = [24,12]
    mpl.rcParams['figure.dpi'] = 240
    fig, ax = plt.subplots()
    x_min = 0.
    x_max = max(t_eval)
    y_min = -1.
    y_max = max(actual_data_df['y'].to_numpy().max()*1.1,pred_df_density['value'].to_numpy().max()*1.1)

    # individual samples
    [ax.plot(t_eval, pred_df[sample_col], linestyle='solid',color='blue',lw=0.25,alpha=0.8,zorder=1,label='_sample_'.format(sample_col=sample_col)) for sample_col in pred_sample_col_list]

    # 5-95 quantile range
    ax.fill_between(t_eval,pred_df['quantile_025'], pred_df['quantile_975'], alpha=0.25,color='green',zorder=0,label='95% Prediction Interval') 

    # median
    ax.plot(t_eval, pred_df['quantile_500'], linestyle='solid',color='white',alpha=1.,lw=6.,zorder=2,label='_median')
    ax.plot(t_eval, pred_df['quantile_500'], linestyle='solid',color='green',alpha=1,lw=4.,zorder=3,label='Median')

    # actual data
    ax.scatter(actual_data_df['t'], actual_data_df['y'], label='_actual_data',color='white',lw=5.,zorder=4)
    ax.scatter(actual_data_df['t'], actual_data_df['y'], label='Actual Data',color='coral',lw=3.5,zorder=5)

    # train data
    ax.scatter(train_data_df['t'], train_data_df['y'], label='_traing_data',color='white',lw=5.,zorder=6)
    ax.scatter(train_data_df['t'], train_data_df['y'], label='Train Data',color='black',lw=3.5,zorder=7)


    plt.title('Density Plot - Individual Samples View | {sample_cnt} Samples'.format(sample_cnt=sample_count))
    plt.xlabel('t')
    plt.ylabel('f(t)')
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min,x_max)
    ax.grid()
    plt.legend()
    plt.show()


#%%
# main script
# initial/config data for script
config.update("jax_enable_x64", True)

random_seed_int = 0
rng = random.PRNGKey(random_seed_int)
np.random.seed(random_seed_int)

key, subkey = random.split(rng)
py_file_path = os.path.dirname(os.path.abspath(__file__))
jax_model_weights_file_name = 'model_weights_jax.pkl'
jax_apply_fun_file_name = 'model_apply_fun_jax.pkl'
initial_guess = jnp.array([25.,0.1])
jnp.set_printoptions(suppress=True)


# nn params for target ode function's parameters (count)
param_count = 2
# nn inputs should be fixed given the context of use for the neural network
nn_inputs = jnp.ones((2,param_count)) * 1e-7
nn_dropout_value = 0.1
nn_units_per_layer = 2048
primer_step_size = 2e-3
traing_step_size = 2e-5
primer_train_iter = 50000
train_iter = 5000
sample_count = 200


# train and actual data params
true_func_initial_Y0 = jnp.array([27.3])
true_func_params = jnp.array([0.041])

training_delta_modifier = 1.1
add_training_noise_func_params = {'offset': training_delta_modifier, 'scale': training_delta_modifier}
training_size_ratio = 0.2
max_training_range_t = 75
min_training_range_t = 0
training_element_cnt_t = int(max_training_range_t * training_size_ratio)

actual_t = jnp.arange(min_training_range_t, max_training_range_t + 1).astype(jnp.float64)

# batch shape:= (t, y)
actual_data_batch = generate_data_batch(true_func_initial_Y0, actual_t, true_func_params, diff_eq)
training_data_batch = slice_actual_data_batch(actual_data_batch, training_element_cnt_t, add_noise_func=add_noise, add_noise_params=add_training_noise_func_params)
# create pandas dataframes of training and actual data
actual_data_df, train_data_df = [pd.DataFrame({'t': t_,'y': y_}) for t_,y_ in [actual_data_batch, training_data_batch]]



#%%
# define nn model and params
init_fun, apply_fun = stax.serial(
    Dense(nn_units_per_layer), elementwise(mish), Dropout(nn_dropout_value),
    Dense(param_count), elementwise(elu,**{'offset':1.})
)

in_shape = ((1,param_count))
out_shape, net_params = init_fun(rng, in_shape)


#%%
# prime params for actual training
primer_batch = jnp.array([jnp.arange(0,len(initial_guess)),initial_guess])
opt_init, opt_update, get_params = optimizers.adamax(step_size=primer_step_size)
opt_state = opt_init(net_params)
prime_params_func_params = {
    'initial_guess': initial_guess,
    'batch': primer_batch,
    'net_params': net_params,
    'train_iter': primer_train_iter,
    'nn_inputs': nn_inputs,
    'key': key
}
net_params, key = prime_params(**prime_params_func_params)

#%%
# train/optimize params
opt_init, opt_update, get_params = optimizers.adamax(step_size=traing_step_size)
opt_state = opt_init(net_params)
train_params_func_params = {
    'batch': training_data_batch,
    'net_params': net_params,
    'opt_state': opt_state,
    'train_iter': train_iter,
    'nn_inputs': nn_inputs,
    'key': key
}
net_params, key = train_params(**train_params_func_params)
opt_state = opt_init(net_params)



#%%
# get model samples
thetas, theta_list, pred_list = sample_model(opt_state, sample_count, actual_data_df['t'].to_numpy(), nn_inputs, key)

#%%
# organize distribution data and archive
theta_df, pred_df, pred_df_density, pred_sample_col_list = convert_samples_to_dataframe(thetas, pred_list, param_count, sample_count, py_file_path)

#%%
# save weight data via pickle
print('Savings Weights | Filename: {file_name}'.format(file_name=jax_model_weights_file_name))
trained_params = optimizers.unpack_optimizer_state(opt_state)
pickle.dump(trained_params, open(os.path.join(py_file_path, jax_model_weights_file_name), "wb"))
# load from saved weight data files
trained_params = pickle.load(open(os.path.join(py_file_path, jax_model_weights_file_name), "rb"))
trained_opt_state = optimizers.pack_optimizer_state(trained_params)


#%%
# plot 1 - main density plot
main_density_plot(actual_data_df['t'].to_numpy(), sample_count, pred_df, actual_data_df, train_data_df, pred_df_density)

#%%
# plot ode parameter distributions
for theta_col in range(param_count):
    xlabel = 'f(0)' if theta_col == 0 else 'Growth Rate' if theta_col == 1 else None
    plot_theta_density(theta_df,theta_col,xlabel)

#%%
# plot contour
plot_contour(actual_data_df['t'].to_numpy(), actual_data_df, pred_df, pred_df_density, sample_count)
#%%
# plot individual samples
plot_samples(actual_data_df['t'].to_numpy(), actual_data_df, pred_df, pred_sample_col_list)

# %% - end of script
