import numpy as np
import math
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interp
import os, sys
from scipy.special import erf, expi
from mpi4py import MPI
from sklearn.mixture import GaussianMixture as GMM
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

#import warnings
#warnings.filterwarnings('error')

Nmax = 15

def Calc_lims(arr1, ind):

        Nbins = 10                          # Now calculate the 68% confidence levels from all the samples, for outputting/plotting
        x = np.array([0.]*Nbins)
        NLevs = 50
    
        if ind==3:
            arr = arr1
        else:
            arr = arr1[arr1>0.]             # All of these are supposed to be >0, which is why we're taking logs below
            if float(len(arr))/len(arr1) < 0.9:
                print(">10 percent of the points are <= 0. Something's probably wrong.", ind)
            
        Nenclose = 0.68*len(arr)
        
        if ind==3:
            y, x11 = np.histogram(arr,Nbins)                # For w1 we don't take log
        else:
            y, x11 = np.histogram(np.log10(arr),Nbins)      # For V and a we do take log
        y = np.array(y)
        x11 = np.array(x11)
        bin_width = x11[1]-x11[0]
        x[0] = (x11[0] + x11[1])/2.
        for j in range(1,Nbins):
            x[j] = x[0] + j*bin_width       # Positions of the bin centres
        for j in range(NLevs):
            level = np.max(y)*(NLevs-j)/NLevs
            mask = y>level
            Nenc = np.sum(y[mask])
            if Nenc>Nenclose:
                lev_choose = level
                break
        spline = UnivariateSpline(x, y-lev_choose, s=0)
        index = np.where(y==np.max(y))[0]            # The x-value(s) at which you've got most points
        mode = np.median(x[index])                   # So if you have more than 1 possible x values you just take the median
    
        if len(spline.roots())==0:                   # Histogram is maybe very uniform?? Just assume we don't know uncertainty.
            print("Can't find spline roots for array", ind)
            r1=mode
            r2=mode
        else:
            r1 = np.min(spline.roots())
            r2 = np.max(spline.roots())        # So we just take the max and min, which are actually the min range enclosing enough points
        
        if r1 > mode:
            r1=mode
        if r2 < mode:
            r2=mode
        
        if y.any()==len(arr):             # In this case all the points are in the same place -> no errorbar
            r1 = mode
            r2 = mode
                
        return mode, r1, r2
                

def rotation_matrix(axis, theta):       # Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def train_regressors(ii):
    mag, logmasslav, log_P, log_PV, log_acc, log_accV, ang, count, count4x, Distance = np.transpose(np.load("Corr/Correction_NoScreen_"+str(ii)+".npy"))
    # Training data from the sim

    ang = np.nan_to_num(ang)        # These are really supposed to be 0 because nan happens for a=aV

    features = np.array([mag, log_PV, log_accV, count, count4x, Distance])          # Visible features of the training set
    features_T = np.transpose(features)
    
    features_ang = np.array([mag[log_accV != -20], log_PV[log_accV != -20], log_accV[log_accV != -20], count[log_accV != -20], count4x[log_accV != -20], Distance[log_accV != -20]])
    features_ang_T = np.transpose(features_ang)

    if np.isnan(features_T).any() or ~np.isfinite(features_T).any():
        print("There is at least 1 inf or nan in the features array")
        quit()
       
    if np.isnan(log_P).any() or ~np.isfinite(log_P).any() or np.isnan(log_acc).any() or ~np.isfinite(log_acc).any() or np.isnan(ang).any() or ~np.isfinite(ang).any() or np.isnan(logmasslav).any() or ~np.isfinite(logmasslav).any():
        print("There is at least 1 inf or nan in the signals array")
        quit()
       
    scaler = preprocessing.StandardScaler().fit(features_T)     # Scale the features array (common to all of the below)
    scaler_ang = preprocessing.StandardScaler().fit(features_ang_T)
    X_transformed = scaler.transform(features_T)
    X_transformed_ang = scaler_ang.transform(features_ang_T)
    
    clf_logP = MLPRegressor(activation='logistic', alpha=0.0001, batch_size='auto',
            beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=400, learning_rate='adaptive',
            learning_rate_init=0.001, max_iter=200, momentum=0.9,
            n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
            random_state=(None if SEED is None else SEED*1000+RANK), shuffle=True, solver='adam', tol=0.0001,
            validation_fraction=0.1, verbose=False, warm_start='True')
    clf_logP_fit = clf_logP.fit(X_transformed, log_P)

    clf_logacc = RandomForestRegressor(bootstrap='True', criterion='squared_error', max_depth=20,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, 
            min_samples_leaf=80, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,        # 700
            oob_score=False, random_state=(None if SEED is None else SEED*1000+RANK), verbose=0, warm_start=False)
    clf_logacc_fit = clf_logacc.fit(X_transformed, log_acc)
    
    clf_ang = RandomForestRegressor(bootstrap='True', criterion='squared_error', max_depth=20,
            max_features='log2', max_leaf_nodes=None,
            min_impurity_decrease=0.0, 
            min_samples_leaf=2, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,        # 1000
            oob_score=False, random_state=(None if SEED is None else SEED*1000+RANK), verbose=0, warm_start=False)
    clf_ang_fit = clf_ang.fit(X_transformed_ang, ang[log_accV != -20])
    
    clf_logmasslav = RandomForestRegressor(bootstrap='True', criterion='squared_error', max_depth=80,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, 
            min_samples_leaf=20, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,        # 1000
            oob_score=False, random_state=(None if SEED is None else SEED*1000+RANK), verbose=0, warm_start=False)
    clf_logmasslav_fit = clf_logmasslav.fit(X_transformed, logmasslav)
    
    return scaler, scaler_ang, clf_logP_fit, clf_logacc_fit, clf_ang_fit, clf_logmasslav_fit


def ScreeningFunc(arr, data_start, data_end):
    
    recv_buffer = np.zeros([data_end-data_start, 15+3*Nmax])        # This is the number of quantities we're printing out for each gal
    
    for i in range(len(arr[:,1])):          # This is over all the gals that have been assigned to this proc
    
        #print(len(arr[:,1]), i)          # This should go up to the right number for the final thread
        
        ra1, dec1, R1, mag, Reff, phi_sers, phi_elpet, w1_r, w1_i, log_vvir_med, log_vvir_err, log_rs_med, log_rs_err, log_rhors_med, log_rhors_err = arr[i,0]*np.pi/180., arr[i,1]*np.pi/180., arr[i,2], arr[i,3], arr[i,4], arr[i,5], arr[i,6], arr[i,7], arr[i,8], arr[i,9], arr[i,10], arr[i,11], arr[i,12], arr[i,13], arr[i,14]
        # This is what you actually need to do the calculation here; phi is angle for finding a_perp

        if R1 >= 250.:          # Only considering testpoints which are close, for greatest resolution and chance of seeing unscr things
            print("I don't think this should be happening", i, R1)

        
        # TESTPTS                           # Now we define a bunch of things that depend on the test galaxy but are the same for all FDSs, ie distances
        x1t = R1*np.cos(dec1)*np.cos(ra1)   # These angles were converted to rad when they were read in (as have Delta_RA and Delta_DEC)
        y1t = R1*np.cos(dec1)*np.sin(ra1)   # Cartesian positions for the Lavaux part; Mpc
        z1t = R1*np.sin(dec1)

        V_out_TOT = np.zeros(repeat)        # Separate for each test point (reset) -- combined from uncertainties and correction factors
        a_out_TOT = np.zeros(repeat)

        w1_mock = np.zeros(repeat)          # Over all the FDSs for this gal

        # SOURCE POINTS
        argument = np.cos(dec1) * np.cos(AllDec*np.pi/180.) * np.cos(ra1 - AllRa*np.pi/180.) + np.sin(dec1) * np.sin(AllDec*np.pi/180.)
        
        dis2 = 1.e6*(R1*R1 + AllDis*AllDis - 2. * R1 * AllDis * argument)            # No need to take arccos just to then take cos of it
        dis2[dis2<0.]=0.
        dis = np.sqrt(dis2)

        mask = (dis<1000.*Rmax) & (dis>0.01)         # This is where you specify that you want the screening potential within Rmax. 

        count = len(dis[(dis<1000.*Rmax)&(dis>0.01)])     # Number of visible sources within lambda_C
        count4x = len(dis[(dis<4000.*Rmax)&(dis>0.01)])   # Within 4*lambda_C
        
        mask_a = (dis < 4000.*Rmax) & (dis>0.01)            # Now including contributions out to 4*lambda_C because we have exponential cutoff
        
        # LAVAUX POINTS
        disL0 = 1000.*np.sqrt((AllX0 - x1t)*(AllX0 - x1t) + (AllY0 - y1t)*(AllY0 - y1t) + (AllZ0 - z1t)*(AllZ0 - z1t))      # kpc
            
        maskL_a = (disL0 < 4000.*Rmax) & (disL0 > 0.01)     # There's no Rthresh with the Lavaux acceleration anymore; kpc
    
        Nlav = len(AllX0[maskL_a])           # Number of Lavaux points within 4*lambda_C
            
        if Nlav==0:
            print("No Lavaux points around this test point")     # This should never happen
    
        Lindex = np.linspace(0, len(AllX0)-1, len(AllX0), dtype=int)[maskL_a]       # Indices of these Lav masses in the full array
    
        disL = disL0[maskL_a]           # Now all of these are only to the Lav points that we actually care about for this galaxy
        AllX = AllX0[maskL_a]
        AllY = AllY0[maskL_a]
        AllZ = AllZ0[maskL_a]
    
        lavindex_choose = AllLavIndex[mask_a]       # Lavaux indices (in the original lav array) for the sources within 4*lambda_C

        if Rmax >= Rthresh:    # In this case you assume that you have enough points in the interpolated Lavaux file to run on it directly
            maskL = (disL<1000.*Rmax) & (disL>0.01)           # This is now defined on the array of Lav masses that we care about
            phi_scale = 1.
        else:                  # In this case you're going to start with the larger radius that def encloses points, and then scale it down
            maskL = (disL<1000.*Rthresh) & (disL>0.01)
            phi_scale = (Rmax/Rthresh)*(Rmax/Rthresh)         # In this case we scale down by r^2

        #sigma = 0.725 * 1000.      # Gaussian smoothing scale in kpc; this was 0.5 before we expanded the range
        #sigma = 0.242 * 1000.      # 3 sigma within each Lav grid cell
        sigma = 0.363 * 1000.       # 2 sigma within each Lav grid cell
        smoothing_phi = erf(disL[maskL]/math.sqrt(2.)*sigma)
        
        smoothing_a = np.exp(sigma*sigma/(2.*(1000.*Rmax)*(1000.*Rmax))) * (0.5*(1.+erf((disL-sigma*sigma/(1000.*Rmax))/(math.sqrt(2.)*sigma))) - 0.5* (1.-disL/(1000.*Rmax))/(1.+disL/(1000.*Rmax))*(1.-erf((disL+sigma*sigma/(1000.*Rmax))/(math.sqrt(2.)*sigma))) - 2.*disL/(1.+disL/(1000.*Rmax)) /(math.sqrt(2.*math.pi)*sigma) * np.exp(-(disL-sigma*sigma/(1000.*Rmax))*(disL-sigma*sigma/(1000.*Rmax))/(2.*sigma*sigma)))
        
        F5_fac_L = np.exp(-disL/(1000.*Rmax)) * (1. + disL/(1000.*Rmax))          # kpc


        for index in range(repeat):     # FDS loop; indexed by k up to repeat
        
            src_choose = np.random.randint(0, N_SOURCE)         # Choose the source catalog

            src = np.load('Catalog_sources/Sources_'+str(src_choose)+'.npy')          # Sources for the test galaxies
            
            AllMass = 10.**src[:, 0]         # Msun
            AllRvir = 10.**src[:, 1]         # kpc
            AllConc = 10.**src[:, 2]
            
            pot = 3.09e16*3.09e16 * (G * AllMass[mask] / dis[mask]) * np.log(1. + AllConc[mask]*dis[mask]/AllRvir[mask]) / (np.log(1.+AllConc[mask]) - AllConc[mask]/(1.+AllConc[mask]))   # This is the contribution from just the source objects in the galaxy catalog; km^2/s^2

            if np.sum(pot)<=0.:         # Could be sampling from the very low tail above
                log_PV = -10.           # Baseline minimum, which should essentially do nothing
            else:
                log_PV = np.log10(np.sum(pot)/(3.e5*3.e5))      # Unitless

    
            # SECOND calculate a5 from the sources

            smoothing_NFW = -AllConc[mask_a] * (1. + AllConc[mask_a]) * np.exp(-(dis[mask_a] + AllRvir[mask_a]/(AllConc[mask_a]*AllConc[mask_a]))/(1000.*Rmax)) * (np.exp(2.*(dis[mask_a] + AllRvir[mask_a]/(AllConc[mask_a]*AllConc[mask_a]))/(1000.*Rmax)) * (dis[mask_a]/(1000.*Rmax) - 1.)*(AllRvir[mask_a]/AllConc[mask_a] + AllConc[mask_a]*dis[mask_a]) *  expi(-(dis[mask_a] + AllRvir[mask_a]/(AllConc[mask_a]*AllConc[mask_a]))/(1000.*Rmax)) - (1. + dis[mask_a]/(1000.*Rmax))*(AllRvir[mask_a]/AllConc[mask_a]+AllConc[mask_a]*dis[mask_a]) * expi((dis[mask_a] + AllRvir[mask_a]/(AllConc[mask_a]*AllConc[mask_a]))/(1000.*Rmax)) + 2.*np.exp(AllRvir[mask_a]/(1000.*Rmax*AllConc[mask_a]*AllConc[mask_a])) * (AllConc[mask_a]*np.exp(dis[mask_a]/(1000.*Rmax))*dis[mask_a] + (1. + dis[mask_a]/(1000.*Rmax))*(AllRvir[mask_a]/AllConc[mask_a] + AllConc[mask_a]*dis[mask_a]) * (np.euler_gamma + np.log(AllRvir[mask_a]/(1000.*Rmax*AllConc[mask_a]))))) / (2.*(AllRvir[mask_a] + AllConc[mask_a]*AllConc[mask_a]*dis[mask_a])*((1. + AllConc[mask_a])*np.log(1. + AllConc[mask_a]) - AllConc[mask_a]))
            
            acc_x = 3.09e19 * G*AllMass[mask_a]/(dis[mask_a]*dis[mask_a]*dis[mask_a]) * smoothing_NFW * (AllDis[mask_a]*np.cos(AllDec[mask_a]*np.pi/180.)*np.cos(AllRa[mask_a]*np.pi/180.) - R1*np.cos(dec1)*np.cos(ra1))           # All unscr here
            acc_y = 3.09e19 * G*AllMass[mask_a]/(dis[mask_a]*dis[mask_a]*dis[mask_a]) * smoothing_NFW * (AllDis[mask_a]*np.cos(AllDec[mask_a]*np.pi/180.)*np.sin(AllRa[mask_a]*np.pi/180.) - R1*np.cos(dec1)*np.sin(ra1))
            acc_z = 3.09e19 * G*AllMass[mask_a]/(dis[mask_a]*dis[mask_a]*dis[mask_a]) * smoothing_NFW * (AllDis[mask_a]*np.sin(AllDec[mask_a]*np.pi/180.) - R1*np.sin(dec1))    # Array over source pts for this test pt; km/s^2

            a1 = np.sum(acc_x)        # These are just numbers for the given testpt (and FDS)
            a2 = np.sum(acc_y)
            a3 = np.sum(acc_z)
            
            if np.sqrt(a1*a1 + a2*a2 + a3*a3)<=0.:
                log_accV = -20.
            else:
                log_accV = np.log10(np.sqrt(a1*a1 + a2*a2 + a3*a3))     # Doesn't include Lavaux contribution because we don't have that in the sim

            
            # THIRD now that we have the observables from the source catalog we can predict the final variables that we need, from all halos
        
            features = np.array([mag, log_PV, log_accV, count, count4x, R1])      # For the test point that we're predicting
            features_T = np.transpose(features)

            X_test_transformed = scaler.transform(features_T.reshape(1,-1))
            X_test_transformed_ang = scaler_ang.transform(features_T.reshape(1,-1))

            pred_logP = clf_logP_fit.predict(X_test_transformed)[0]            # unitless
            
            if pred_logP < log_PV:          # Even if log_PV = -10 we're assuming that there is some pred_logP
                pred_logP = log_PV

            if log_accV == -20.:
                a1_true, a2_true, a3_true = 0., 0., 0.
                
            else:       # Don't need to bother fitting these if there are no sources within 4*lambda_C because we won't use them
                pred_logacc = clf_logacc_fit.predict(X_test_transformed)[0]             # km/s
                pred_ang = clf_ang_fit.predict(X_test_transformed_ang)[0]
                pred_logmasslav = clf_logmasslav_fit.predict(X_test_transformed)[0]     # Extract number from array
                
                a_true = 10.**pred_logacc                       # After correction; km/s^2
                
                a_orig = math.sqrt(a1*a1 + a2*a2 + a3*a3)       # Now we have to do the rotation; magnitude of a before the correction

                a_bef = [a1*a_true/a_orig, a2*a_true/a_orig, a3*a_true/a_orig]      # Points in the original direction but has length of a_true
           
                ao,b,c = a_bef                  # Components before rotation
                ct = math.cos(pred_ang)         # Cos of angle that you're scattering by
            
                an = (-2.*ao*b*b - 2.*ao*c*c + math.sqrt((2. * ao * b*b + 2.*ao*c*c)**2. - 4.*(ao*ao - ao*ao*ct*ct - b*b*ct*ct - c*c*ct*ct)*(b*b*b*b + 2.*b*b * c*c + c*c*c*c - ao*ao*b*b*ct*ct - b*b*b*b * ct*ct - ao*ao * c*c * ct*ct - 2.*b*b*c*c*ct*ct - c*c*c*c*ct*ct)))/(2.*(ao*ao - ao*ao*ct*ct - b*b*ct*ct - c*c*ct*ct))                     # Example x component of rotated vector
            
                mult_fac = a_true/math.sqrt(an*an + b*b + c*c)          # These are all just numbers
                an *= mult_fac
                b *= mult_fac           # Get the length of the rotated array to the true length
                c *= mult_fac
            
                a_aft = np.dot(rotation_matrix(a_bef, np.random.uniform(0.,2.*math.pi)), [an,b,c])     # Rotate around original by random angle
            
                a1_true, a2_true, a3_true = a_aft       # Components of a5 from all halos down to the resolution limit of the calibrating sim
            
        
            # FOURTH calculate contributions from long-wavelength modes

            lav_choose = np.random.randint(0,2)

            if lav_choose==0:
                AllMass_L0 = AllMass_L_1         # Only need to set mass here because positions already set before FDS loop
            elif lav_choose==1:
                AllMass_L0 = AllMass_L_2         # These are the lavaux masses at (AllX, AllY, AllZ) for this particular realisation
            elif lav_choose==2:
                AllMass_L0 = AllMass_L_3
            elif lav_choose==3:
                AllMass_L0 = AllMass_L_4
            elif lav_choose==4:
                AllMass_L0 = AllMass_L_5
            elif lav_choose==5:
                AllMass_L0 = AllMass_L_6
            elif lav_choose==6:
                AllMass_L0 = AllMass_L_7         # These are refreshed here each FDS calculation so it's fine to modify them below
            elif lav_choose==7:
                AllMass_L0 = AllMass_L_8
            elif lav_choose==8:
                AllMass_L0 = AllMass_L_9
            elif lav_choose==9:
                AllMass_L0 = AllMass_L_10
            else:
                print("Bad", lav_choose)
                quit()
                
            AllMass_L = np.copy(AllMass_L0[maskL_a])    # These are the only Lav masses we care about for this galaxy
            mass_choose = AllMass[mask_a]               # Masses of the source points that we care about; depends on FDS
    
            # Vectorised equivalent of the original per-source np.where loop.
            # Lindex is sorted ascending (it comes from a boolean mask on arange), so
            # searchsorted gives the position of each source's Lavaux cell; sources whose
            # cell is not inside 4*lambda_C are dropped, exactly as the original except-clause did.
            _want = lavindex_choose.astype(np.int64)
            _pos  = np.searchsorted(Lindex, _want)
            _ok   = (_pos < len(Lindex))
            _pos  = np.where(_ok, _pos, 0)
            _ok  &= (Lindex[_pos] == _want)
            np.add.at(AllMass_L, _pos[_ok], -mass_choose[_ok])
            
            if log_accV != -20.:        # Only subtract something from Lavaux if there is some source nearby
                AllMass_L -= 10.**pred_logmasslav/Nlav   # Subtract masslav equally from all points that are within 4xlambda_C; just a number
            
            if np.all(AllMass_L) <= 0.:            # If all the grid elements have been depleted of mass then there's no additional Lavaux contribution
                #print("Final Lav mass is zero: ", rank, i, index)
                V_outL, a1L, a2L, a3L = 0., 0., 0., 0.
                
            else:                                  # If some elements are nonzero then we add the contributions from those
                AllMass_L[AllMass_L < 0.] = 0.     # Can't reduce the masses in the Lavaux cells below 0

                potL = 3.09e16*3.09e16 * G * AllMass_L[maskL] / disL[maskL] * smoothing_phi       # Include new Gaussian smoothing; km^2/s^2

                V_outL = phi_scale * np.sum(potL)           # Potential due to lavaux masses; km^2/s^2
        
                acc_xL = F5_fac_L * 3.09e19 * (G * AllMass_L / disL**3.0) * (AllX - x1t) * smoothing_a      # All unscr here
                acc_yL = F5_fac_L * 3.09e19 * (G * AllMass_L / disL**3.0) * (AllY - y1t) * smoothing_a
                acc_zL = F5_fac_L * 3.09e19 * (G * AllMass_L / disL**3.0) * (AllZ - z1t) * smoothing_a
                # unscr_L is an array over all the Lavaux mass points, like Source, but approximate; km/s^2
            
                a1L = np.sum(acc_xL)          # Don't use the Rthresh scaling for a because we're going out to 4*lambda_C
                a2L = np.sum(acc_yL)
                a3L = np.sum(acc_zL)
                
        
            # FIFTH calculate total acceleration and potential, and hence determine whether this test point is screened

            a1_TOT = a1_true+a1L            # Add Lavaux onto this total from the sources
            a2_TOT = a2_true+a2L
            a3_TOT = a3_true+a3L
            a_out_TOT[index] = np.sqrt(a1_TOT*a1_TOT + a2_TOT*a2_TOT + a3_TOT*a3_TOT)

            if a_out_TOT[index]==0.:
                print("Acceleration is 0: ", rank, i, index, "  ", count4x, log_accV, np.mean(AllMass_L), np.mean(disL), a1L, a2L, a3L)

            Vvir = 10.**np.random.normal(log_vvir_med, log_vvir_err)       # km/s
            V_in = Vvir*Vvir/(3.e5*3.e5)    # unitless

            V_out_TOT[index] = 10.**pred_logP + V_outL/(3.e5*3.e5) + V_in   # Include internal pot too; everything is +ve; unitless

                 
            # SIXTH deal with the signal, assuming all unscreened here; this is still in galaxy and FDS loop

            rs = 10.**np.random.normal(log_rs_med, log_rs_err)
            rho_rs = 10.**np.random.normal(log_rhors_med, log_rhors_err)
        
            a_delta = -a1_TOT*math.sin(dec1)*math.cos(ra1) - a2_TOT*math.sin(dec1)*math.sin(ra1) + a3_TOT*math.cos(dec1)
            a_ra = -a1_TOT*math.sin(ra1) + a2_TOT*math.cos(ra1)         # Components actually needed for signal

            theta = (phi_sers-90.)*math.pi/180.                         # Angle that I actually rotate by in the ipynb, in rad

            a_perp = a_delta*math.cos(theta) + a_ra*math.sin(theta)

            w1_mock[index] = - rho_slope*(3.-rho_slope)/((rho_slope+1.)*(rho_slope+2.)) * a_perp/3.09e16 * delta_G/(G*G) / (4.*math.pi*rho_rs) * (3.*Reff/rs)**rho_slope / (3.*Reff)     # Should be unitless; disk warps down if a5 is up; convert a to kpc
            
            
        mask = (np.isfinite(V_out_TOT)) & (np.isfinite(a_out_TOT)) & (np.isfinite(w1_mock))
        V_out_TOT = V_out_TOT[mask]
        a_out_TOT = a_out_TOT[mask]                 # Remove any pathological cases from the set of FDSs
        w1_mock = w1_mock[mask]

        #print("Fraction of points retained:", float(len(w1_mock))/repeat)

        #frac = float(len(V_out_TOT[V_out_TOT < chi]))/len(w1_mock)        # chi isn't logged here; unscreened frac over all FDSs for this particular gal
        frac = 1.           # All test gals are unscr

        Vmode, r1, r2 = Calc_lims(V_out_TOT, 1)     # Calculate errorbars for log10(V_tot), log10(a_tot) and w1_mock over all FDSs for this gal
        amode, t1, t2 = Calc_lims(a_out_TOT, 2)
        wmode, i1, i2 = Calc_lims(w1_mock, 3)       # We potentially have fewer points than repeat now
   
        if np.all(w1_mock)==0.:           # In this case there's no signal so everything has to be 0 except frac
            print("This galaxy has a w1_mock of 0: ", rank, i, np.min(a_out_TOT), np.median(a_out_TOT), np.max(a_out_TOT))
            recv_buffer[i,:] = [dec1] + [R1] + [0.]*(1+3*Nmax) + [frac] + [w1_r] + [w1_i] + [r1] + [Vmode] + [r2] + [0.]*6
            
        else:                                       # This galaxy has a signal so we proceed with what we're supposed to be calculating
            n_components = np.arange(1,Nmax+1)      # Fit likelihoods with between 1 and Nmax Gaussians
            
            weights, means, covs = np.zeros(Nmax), np.zeros(Nmax), np.zeros(Nmax)
            
            models = [GMM(n, covariance_type='full', random_state=42, max_iter=5000).fit(w1_mock[:, None]) for n in n_components]
            bic = [m.bic(w1_mock[:, None]) for m in models]
            ncomp = np.argmin(bic)+1        # index of 0 would correspond to 1 component
            gmm = GMM(n_components=ncomp, covariance_type='full', random_state=424, max_iter=5000)      # max_iter increased, but can have warning
            gmm.fit(w1_mock[:,None])                  # logprob = gmm.score_samples(x_d[:, None])
            weights0 = gmm.weights_
            means0 = np.ndarray.flatten(gmm.means_)
            covs0 = np.ndarray.flatten(gmm.covariances_)
                
            weights[:ncomp] = weights0
            means[:ncomp] = means0
            covs[:ncomp] = covs0
            
            if rank==0:
                print(i, ncomp)               # This is only for rank 0, while the below is for any rank
            #if ncomp>=14:
                #print("")
                #print(weights, means, covs)       # See what the distribution you're fitting looks like for high number of fitted Gaussians
                #print(w1_mock)
                #print("")
                
            recv_buffer[i,:] = [dec1] + [R1] + [ncomp] + list(weights) + list(means) + list(covs) + [frac] + [w1_r] + [w1_i] + [r1] + [Vmode] + [r2] + [t1] + [amode] + [t2] + [i1] + [wmode] + [i2]              # Fixed number of elements because arrays are padded with zeros to length Nmax
    
    
    return recv_buffer        # This is the slice of the test data that this particular thread is dealing with


#rho_slope = 0.1
rho_slope = 0.5             # CHANGE; negative density slope n. Eg 0.1-1.5. 0.5 should be a reasonable value.
#rho_slope = 1.
#rho_slope = 1.5
    
Rmax_ARG   = float(sys.argv[1])          # lambda_C in Mpc, arbitrary
TAG        = sys.argv[2]                 # output tag
REPEAT_ARG = int(sys.argv[3])            # number of MC realisations
# Replicate seed comes from the environment, NOT argv: sys.argv[4] is already the producer's
# "use only the first N test points" debug switch (line ~513) and passing a seed there silently
# truncates the sample to SEED galaxies.
SEED       = int(os.environ['REPLICATE_SEED']) if os.environ.get('REPLICATE_SEED') else None
ii = 19                                  # correction regressors frozen at lambda_C = 8 Mpc (documented approximation)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
RANK = rank

# Seeded replicate: every rank draws an independent but reproducible stream, and trains its own
# regressor as before (random_state = SEED*1000+rank, not a single shared value -- sharing one
# would remove the between-rank regressor variance that the unseeded run carries).
if SEED is not None:
    np.random.seed(SEED*100003 + rank)

AllX0, AllY0, AllZ0, AllMass_L_1 = np.transpose(np.load('/users/hdesmond/LowFreq/LowFreq_fin1_7900_400.npy'))           # Mpc, Msun
AllMass_L_2 = np.load('/users/hdesmond/LowFreq/LowFreq_fin1_8000_400.npy')       # Only the one above containts positions, the rest are 1D with mass only

scaler, scaler_ang, clf_logP_fit, clf_logacc_fit, clf_ang_fit, clf_logmasslav_fit = train_regressors(ii)     # This training can be done anywhere

N_SOURCE = 200


'Loop over test points to calculate proxies'
    
G = 4.52e-39                # Msun, kpc, s
Sig_crit = 1.37e8           # In Msun/kpc^2; for calculating the central density

repeat = REPEAT_ARG         # ATTENTION Number of FDSs; 20, 200, 500, 1000, 2000

delta_G = 1.*G              # In units of G

Rmax_min = 0.4      # Mpc
Rmax_max = 8.
Rmax_num = 20
Rmax_step = (Rmax_max - Rmax_min)/float(Rmax_num-1)

Rmax = Rmax_ARG      # lambda_C, from argv

chi = 1.5*(Rmax/3.2e3)*(Rmax/3.2e3)      # unitless; H-S; try variations of this by 1 or 2 OOM

AllRa, AllDec, AllDis, AllLavIndex = np.transpose(np.load('Catalog_sources/NSA_sources_plus.npy'))        # From prepare_sources.py & augment_sources.py

Nsrc = len(AllRa)               # All NSA sources

x1 = AllDis*np.cos(AllDec*np.pi/180.)*np.cos(AllRa*np.pi/180.)      # Cartesian positions of all sources (as opposed to x1t for testpts)
y1 = AllDis*np.cos(AllDec*np.pi/180.)*np.sin(AllRa*np.pi/180.)      # This is where unscr_out is evaluated, which you need to know for the interpolation
z1 = AllDis*np.sin(AllDec*np.pi/180.)

Rthresh = 3.5           # This is reduced because we're now using 400 Lav points across the range rather than 100;  10, 5

#unscr, unscr_out = np.transpose(np.load('unscr/unscr_'+str(ii)+'.npy'))        # Number of lines = number of source gals

#unscr_L = np.zeros(len(AllX0))         # Interpolate from the above
#fn = interp.LinearNDInterpolator(np.array((x1,y1,z1)).T, np.array((unscr_out)).T, fill_value=1.)

#for l in range(len(AllX0)):
    #unscr_L[l] = fn(np.array([AllX0[l], AllY0[l], AllZ0[l]]).T)         # Search at the position of this testpt to get unscr_L frac (within 60)


testpts = np.load('warp_halos.npy')  #RA, Dec, Dist, Mag, Reff, phi_sers, phi_elpet, w1_r, w1_i, vvir_med, vvir_err, rs_med, rs_std, rhors_med, rhors_std

import os as _os
if _os.environ.get('FULLSAMPLE', '0') != '1':
    TOPIDX = np.load('top5_idx.npy').astype(int)
    testpts = testpts[TOPIDX, :]
if len(sys.argv)>4:
    testpts = testpts[:int(sys.argv[4]), :]
if rank==0:
    print("Total number of test points: ", len(testpts))

nLs = int(np.ceil(len(testpts) / float(size)))       # Number of lines per file for given thread

if rank==0:
    print("Number of test points per proc: ", nLs)

data_start = rank*nLs
data_end = (rank+1)*nLs

if rank==size-1:
    data_end = len(testpts)
        
thread_data = testpts[data_start:data_end, :]             # So this is all of Ra, Dec, Dist but only for some number of gals
        
outp = ScreeningFunc(thread_data, data_start, data_end)

np.savetxt('Partial/out_warp_FULL_'+TAG+'_'+str(rank)+'.dat', outp, fmt='%.8f')         # The 0p5 here is for rho_slope


buff = np.zeros(1)              # Now wait for everything to finish before combining the partial arrays

if rank==0:
    for i in range(1, size):
        comm.Recv(buff, source=i)
else:
    comm.Send(buff, dest=0)

if rank == 0:
    string = 'cat `find ./Partial/ -name "out_warp_FULL_'+TAG+'_*.dat" | sort -V` > Signals_warp_FULL_'+TAG+'.dat'
    os.system(string)
    string = 'rm Partial/out_warp_FULL_'+TAG+'_*.dat'
    os.system(string)
