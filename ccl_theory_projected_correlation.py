#initialization
from pyccl import Cosmology
from pyccl.correlations import correlation_pi_sigma
from scipy.integrate import quad
from numpy import savetxt
from numpy import zeros

def ccl_theory_projected_correlation(Om_c,Om_b,h_c,A_sc,n_sc,a,b,lp,ls,bin):
    """
    Computes the correlation function along line of sight, for given parameters, from theory
    Om_c (float) = Density parameter for dark matter
    Om_b (float) = Density of Baryonic matter
    h_c (float) = Hubble paramter H0/100
    A_sc (Exponent) = Amplitude/Proportionality constant of Zeldovich spectrum (P=A_sc*l^n_sc)
    n_sc (float) = value of index in the power spectrum (scale dependence of laws)
    a (float) = scale factor
    b (float) = growth rate/galaxy bias (Metric for redshift based distortion)
    lp (integer) = upper bound of r_parallel in computation (does not take h^-1 inputs)
    ls (integer) = upper bound of r_perpendicular in computation (does not take h^-1 inputs)
    bin (list - optional) = pass an of the bins you wish to calculate the function for

    Returns:
    fnc (array) = array of calculated values of correlation function along line of sight 
    rl (array) = array of corresponding r_perpendicular values (in h^-1 mpc)
    theory_projected_correlation_function.csv = csv file containing the values of fnc
    r_perpendicular.csv = csv file containing the values of rl
    
    Default values (partly from planck data): ccl_theory_projected_correlation(0.267,0.049,0.67,2.1e-9,0.965,0.5,1,1000,1000)
    """

    #defining the cosmology based on the input parameter
    cosmo = Cosmology(Omega_c=Om_c, Omega_b=Om_b, h=h_c, A_s=A_sc, n_s=n_sc)
    #defining the sizes of the output arrays
    fnc = zeros(ls-1)
    sl = zeros(ls-1)
    #setting initial value of s to 0, for the integral, we iterate over s later
    s = 0
    #defining integrand (integrating over all r_parralel)
    def integr (p):
        return correlation_pi_sigma(cosmo,pi= p,sigma= s,a=a,beta=b)
    # iterating the aforementioned integral over all r_perpendicular
    if len(bin) == 0:
        for i in range(1,int(ls)): #makes no sense to check correlation at r = 0, starting point irrelevant as long as it is small
            s = i*h_c #converting to h^-1
            t,_ = quad(integr,0.1,h_c*lp) #converting to h^-1
            fnc[i-1]= t
            sl[i-1] = s
    else:
        for i in range(len(bin)): #makes no sense to check correlation at r = 0, starting point irrelevant as long as it is small
            s = bin[i]*h_c #converting to h^-1
            t,_ = quad(integr,0.1,h_c*lp) #converting to h^-1
            fnc[i-1]= t
            sl[i-1] = s
    #making csv files
    savetxt("theory_projected_correlation_function.csv", fnc , delimiter=", ", newline=", ", fmt='%.4e')
    savetxt("r_perpendicular.csv", sl , delimiter=", ", newline=", ", fmt='%.4e')

    return(fnc,sl)
