import torch
import torch.nn as nn
import pandas as pd
import math
import random
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

t = 0.5
seed = 42
# MAX_EPOCHS = int(1e6)

# problem setup: min( ||p*a - t||^2 + \lambda*\sum{(p_i)*(1-p_i)*a_i^2})

# choose lambda
def get_regularization(n, p, a, lmbda=1.0):
    reg = torch.sum(p * (1-p) * torch.pow(a, 2))
    reg = lmbda * reg
    return reg


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("Seeded everything: {}".format(seed))


set_seed(seed)

# run GD and return results
def subset_sum(num_samples=10, lmbda=0, lr=0.0001, p_init='uniform', optim_algo='Adam'):
    a = torch.zeros(num_samples)
    p = nn.Parameter(torch.Tensor(a.size()))
    a.requires_grad = False
    p_list = []
    tot_loss_list = []
    subset_loss_list = []
    epoch_list = []

    if optim_algo == 'Adam':
        optimizer = optim.Adam(
            [p],
            lr=0.01,
            weight_decay=0)
    else:
        optimizer = optim.SGD(
            [p],
            lr=lr,
            weight_decay=0)

    # initialize a as uniform [-1, 1]
    nn.init.uniform_(a, a=-1, b=1)
    if p_init == 'uniform':
        nn.init.uniform_(p, a=0, b=1)
    else:
        nn.init.normal_(p, mean=0.5)
        p.data = torch.clamp(p.data, 0.0, 1.0)

    # need to increase T for each n
    MAX_EPOCHS = int(1e6*math.log(num_samples,10))
    CONVERGED_FLAG = False
    prev_loss = np.inf
    for num_iter in range(MAX_EPOCHS):
        optimizer.zero_grad()
        loss = (t - torch.sum(p*a))**2 + get_regularization(num_samples, p, a, lmbda=lmbda)
        loss.backward()
        optimizer.step()
        # commenting out the part where we remember iterates. These can be far too many.
        # loss we care out is not the total loss. Just the subset sum loss
        ## subset_loss_list.append(((t - torch.sum(p*a))**2).item())
        # also remember total loss for stopping criterion
        ## tot_loss_list.append(loss.item())
        ## epoch_list.append(num_iter)
        ## p_list.append(p.data)

        if num_iter % 1000 == 0 and num_iter != 0:
            print("Iteration={} | Loss={}".format(num_iter, loss))
        p.data = torch.clamp(p.data, 0.0, 1.0)

        if num_iter > 5 and loss.item() == prev_loss:
            print("Iteration={} | Converged".format(num_iter))
            CONVERGED_FLAG = True
            break

        prev_loss = loss.item()

    # by the recent paper, error should be O(1/n^logn)
    min_error = 1.0/(num_samples**np.log(num_samples))
    # print("Minimum Error = {}".format(min_error))
    # results_df = pd.DataFrame({'epoch': epoch_list, 'loss': subset_loss_list})
    results_df = pd.DataFrame()

    return min_error, p, results_df, CONVERGED_FLAG

# returns number of vertices which are fractional
def get_num_frac(x, num_samples):
    num_middle = torch.sum(torch.gt(x,
                                    torch.ones_like(x)*0) *
                           torch.lt(x,
                                    torch.ones_like(x*(1)).int())).item()
    return 1.0*num_middle/num_samples


# returns l2 distance to the nearest vertex
def get_dist_to_vertex(x):
    rounded_x = torch.gt(x, torch.ones_like(x)*0.5).int().float()
    return torch.norm(x-rounded_x).item()


if __name__ == "__main__":
    avg_error_ratios = []
    NUM_LMBDAS = 10
    NUM_RANGE = [10, 1e2, 1e3, 5e3, 1e4, 1e5]
    df_list = []
    for num_samples in NUM_RANGE:
        print("Num Samples = {}".format(num_samples))
        error_ratio_list = []
        min_error_list = []
        error_list = []
        lmbda_list = []
        num_frac_list = []
        dist_to_vertex_list = []
        converged_flag_list = []
        min_lmbda = 1/(num_samples**math.log(num_samples))
        max_lmbda = 1.0
        lmbda_range = np.linspace(min_lmbda, max_lmbda, NUM_LMBDAS, endpoint=True)

        for lmbda in lmbda_range:
            num_samples = int(num_samples)
            min_error, p_final, results_df, CONVERGED_FLAG = subset_sum(num_samples, lmbda)

            final_error = results_df.tail(1).loss.item()
            error_list.append(final_error)
            min_error_list.append(min_error)
            lmbda_list.append(lmbda)
            num_frac = get_num_frac(p_final, num_samples)
            num_frac_list.append(num_frac)
            dist_to_vertex = get_dist_to_vertex(p_final)
            dist_to_vertex_list.append(dist_to_vertex)
            error_ratio = 1.0*final_error/min_error
            error_ratio_list.append(error_ratio)
            converged_flag_list.append(CONVERGED_FLAG)

            print("\n\n\n-------------------------------------------------------------------------------------------------------------------")
            print("Lambda = {} | Converged = {} | Error_ratio={} | Final error={} | Minimum error={} | num_frac={} | dist_to_veretx={}".
                  format(lmbda, CONVERGED_FLAG, error_ratio, final_error, min_error, num_frac, dist_to_vertex))
            print("-------------------------------------------------------------------------------------------------------------------\n\n\n")

        num_samples_list = [num_samples for x in range(len(lmbda_list))]
        results_df = pd.DataFrame({"num_samples": num_samples_list, "lambda": lmbda_list, "error": error_list,
                                   "num_frac": num_frac_list, "dist_to_vertex": dist_to_vertex_list,
                                   "min_error": min_error_list, "error_ratio": error_ratio_list,
                                   "converged": converged_flag_list})

        results_df['lambda'] = results_df['lambda'].to_numpy()
        results_df['error'] = results_df['error'].to_numpy()
        results_df['num_frac'] = results_df['num_frac'].to_numpy()
        results_df['min_error'] = results_df['min_error'].to_numpy()
        df_list.append(results_df)
        # print for each lambda the non 0-1 for each number of samples
        plt.figure()
        ax = results_df[results_df['converged'] == True].plot(x="lambda", y="num_frac", kind="scatter", color='blue',
                                                              title="Num samples={}".format(num_samples))
        results_df[results_df['converged'] == False].plot(ax=ax, x="lambda", y="num_frac", kind="scatter", color='red',
                                                          title="Num samples={}".format(num_samples))
        plt.savefig("results/lambda_vs_num_frac_num_samples_{}.pdf".format(num_samples),format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
        # plot for each lambda the distance from the vertex for the minimum
        plt.figure()
        ax = results_df[results_df['converged'] == True].plot(x="lambda", y="dist_to_vertex", kind="scatter", color="blue",
                                                              title="Num samples={}".format(num_samples))
        results_df[results_df['converged'] == False].plot(ax=ax, x="lambda", y="dist_to_vertex", kind="scatter", color="red",
                                                          title="Num samples={}".format(num_samples))
        plt.savefig("results/lambda_vs_dist_to_vertex_num_samples_{}.pdf".format(num_samples), format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.figure()
        ax = results_df[results_df['converged'] == True].plot(x="lambda", y="error", kind='scatter', color="blue",
                                                              title="Num samples={}".format(num_samples))
        results_df[results_df['converged'] == False].plot(ax=ax, x="lambda", y="error", kind='scatter', color="red",
                                                          title="Num samples={}".format(num_samples))
        # plot the minimum error for different lambdas for each number of samples
        results_df.plot(x="lambda", y="min_error", ax=ax)
        plt.savefig("results/lambda_vs_error_num_samples_{}.pdf".format(num_samples), format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)

    combined_results_df = pd.concat(df_list)
    combined_results_df.to_csv("results/combined_results.csv", index=False)
