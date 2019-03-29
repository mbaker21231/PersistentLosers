import numpy as np
from scipy.stats import norm, uniform

def bm_basic(n=1000, x=0, mu=0, sigma=1, dt=.1):
    
    """
    This function generates a basic brownian motion with n observations.
    
    The x is the starting point, mu is the mean of the process,
    the sigma is the standard deviation of the process. 
    
    The step size is set at .1, while n is the number of draws. 
    
    """
    vals = np.zeros((n, 1))

    for k in range(n):
        x = x + mu*dt + norm.rvs(scale=sigma**2*dt)
        vals[k] = x
        
    return vals

def bm_1switch(n=1000, x=0, mus=[0, -.1], sigmas=[1, 1.2], cut=-1, dt=.1):
    
    """ This function generates a brownian motion that transitions 
    through a regime. The first set of parameters apply so long 
    as the process is above the cutpoint, and the second set 
    of parameters applies below.  Otherwise parameters are as defined 
    by the simple brownian motion. function bm_basic."""
    
    vals = np.zeros((n, 1))

    for k in range(n):
        if x >= cut: 
            x = x + mus[0]*dt + norm.rvs(scale=sigmas[0]**2*dt)
            vals[k] = x
        else:
            x = x + mus[1]*dt + norm.rvs(scale=sigmas[1]**2*dt)
            vals [k] = x
    
    return vals

def bm_2switch(n=1000, x=0, mus=[0, -.1], sigmas=[1, 1.2], cut=[-1, -2], dt=.1):
    
    """ This function generates a brownian motion that transitions 
    through a middle regime where risk-taking behavior occurs. The first set of parameters apply so long 
    as the process is above the cutpoint, and below the second cutpoint. The second set of parameters applies in between. the second set 
    of parameters applies below.  Otherwise parameters are as defined 
    by the simple brownian motion. function bm_basic."""
    
    vals = np.zeros((n, 1))

    for k in range(n):
        if x >= cut[0] or x < cut[1]: 
            x = x + mus[0]*dt + norm.rvs(scale=sigmas[0]**2*dt)
            vals[k] = x
        else:
            x = x + mus[1]*dt + norm.rvs(scale=sigmas[1]**2*dt)
            vals [k] = x
    
    return vals

def bm_basic_d(n=1000, x=0, mu=0, sigma=1, dt=.1, dr=.05):
    
    """
    This function generates a basic brownian motion with n observations, but now there is a flow probability of death, in which
    event the process starts over again at 0. 
    
    The x is the starting point, mu is the mean of the process,
    the sigma is the standard deviation of the process. 
    
    The step size is set at .1, while n is the number of draws. 
    
    The flow probability of death is dt*dr. 
    
    """
    vals    = np.zeros((n, 1))
    devents = np.zeros((n, 1))

    for k in range(n):
        if uniform.rvs() < dt*dr:
            x = 0
            devents[k] = 1
        else:
            x = x + mu*dt + norm.rvs(scale=sigma**2*dt)
        vals[k] = x
        
    return vals, devents


def bm_1switch_d(n=1000, x=0, mus=[0, -.1], sigmas=[1, 1.2], cut=-1, dt=.1, dr=.05):
    
    """ This function generates a brownian motion that transitions 
    through a regime. The first set of parameters apply so long 
    as the process is above the cutpoint, and the second set 
    of parameters applies below.  Otherwise parameters are as defined 
    by the simple brownian motion. function bm_basic. But now, we will have a death rate as in the basic brownian motion..."""
    
    vals    = np.zeros((n, 1))
    devents = np.zeros((n, 1))

    for k in range(n):
        if uniform.rvs() < dt*dr:
            x = 0
            devents[k] = 1
        elif x >= cut: 
            x = x + mus[0]*dt + norm.rvs(scale=sigmas[0]**2*dt)
            vals[k] = x
        else:
            x = x + mus[1]*dt + norm.rvs(scale=sigmas[1]**2*dt)
            vals [k] = x
    
    return vals, devents

def bm_2switch_d(n=1000, x=0, mus=[0, -.1], sigmas=[1, 1.2], cut=[-1, -2], dt=.1, dr=.05):
    
    """ This function generates a brownian motion that transitions 
    through a middle regime where risk-taking behavior occurs. The first set of parameters apply so long 
    as the process is above the cutpoint, and below the second cutpoint. The second set of parameters applies in between. the second set 
    of parameters applies below.  Otherwise parameters are as defined 
    by the simple brownian motion. function bm_basic."""
    
    vals    = np.zeros((n, 1))
    devents = np.zeros((n, 1))

    for k in range(n):
        if uniform.rvs() < dt*dr:
            x = 0
        elif x >= cut[0] or x < cut[1]: 
            x = x + mus[0]*dt + norm.rvs(scale=sigmas[0]**2*dt)
            vals[k] = x
        else:
            x = x + mus[1]*dt + norm.rvs(scale=sigmas[1]**2*dt)
            vals [k] = x
    
    return vals

def lrd(fun, obs=1000, **kwargs):
    
    """This function simulates the long-run distribution of a brownian motion
       It basically just runs the function provided in the argument for n 
       periods, obs times, and then takes the last observation as a
       representative of the distribution. Keyword arguments have to be 
       passed along as follows: **{"n":200, "mus":[0,1]}"""
    
    end_vals=np.zeros((obs, 1))
    for t in np.arange(obs):
        x = fun(**kwargs)
        end_vals[t] = x[0][-1]
    
    return end_vals
    

def bmv_basic(n=1000, runs= 100, x=0, mu=0, sigma=1, dt=.1):
    
    """
    This function generates a basic brownian motion with n observations, 
    but creates 100 "runs" number of runs in parallel. The idea is to
    figure out how to speed the simulation of multiple brownian motions.
    
    The x is the starting point, mu is the mean of the process,
    the sigma is the standard deviation of the process. 
    
    The step size is set at .1, while n is the number of draws. 
    
    """
    vals = np.zeros((runs, n))

    for k in range(n):
        x = x + mu*dt + norm.rvs(scale=sigma**2*dt)
        vals[k] = x
        
    return vals