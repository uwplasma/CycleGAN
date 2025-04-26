import numpy as np
from simsopt.mhd import Boozer
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.special import ellipe
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# Returns a magnetic field's QI deviation on a flux surface
def QuasiIsodynamicResidual(vmec,snorms,weights=None,
                nphi=141,nalpha=27,nBj=51,
                mpol=18,ntor=18,
                nphi_out=2000,
                arr_out=True):
    """
    Calculate the deviation in omnigenity for a VMEC equilibrium at some normalized toroidal flux.
    vmec        -   VMEC object
    snorms      -   Flux surfaces at which the penalty will be calculated
                    For low modes, I generally choose 4-5 flux surfaces, evenly spaced
    weights     -   Relative weights of flux surfaces
                    I would recommend leaving this as "None"
    nphi        -   Number of points along measured along each well
                    This should be >100, and always be >nBj, and should be increased as the number
                    of VMEC Fourier modes is increased
    nalpha      -   Number of wells measured
                    I usually set this to 51, and then 76 with more VMEC Fourier modes
    nBj         -   Number of bounce points measured
                    Should be no greater than ~2/3 of nphi
    mpol        -   Poloidal modes in Boozer transformation
                    Should be ~20 for low VMEC Fourier modes, ~40 for more VMEC Fourier modes
    ntor        -   Toroidal modes in Boozer transformation
                    Should be ~20 for low VMEC Fourier modes, ~40 for more VMEC Fourier modes
    arr_out     -   If True, returns (nphi_out*nalpha) values, each of which is the difference
                    between B and B_{QI} at a given (\alpha, \varphi).
                    If False, returns (nalpha) values, each of which is ∫(B and B_{QI})^2 d\phi
                    for a different fieldline
                    In general, this should be set to True when optimizing a low number of modes,
                    as it will help with convergence. This comes at a significant cost to memory,
                    so when optimizing with more modes, having this as True cause the optimization
                    to crash.
    """
    ################################################################
    ############################ SETUP #############################
    ################################################################
    vmec.run()
    try:
        ns = len(snorms)
    except:
        snorms = [snorms]
        ns = 1
    try:
        weights = np.sqrt(weights)
    except:
        weights = np.ones(ns)
    
    # The size of the output penalty array depends on arr_out
    if arr_out == True:
        out = np.zeros((ns,nalpha,nphi_out))
    else:
        out = np.zeros((ns,nalpha))

    # Construct Boozer object
    boozer = Boozer(vmec,mpol,ntor,verbose=False)
    boozer.register(snorms)
    boozer.run()

    # The 2D array of toroidal angles
    nfp = vmec.wout.nfp
    ####################################################################################
    ####################################### NOTE #######################################
    ####################################################################################
    # phimin and phimax are the locations of the first and second maxima of the well
    # respectively. The location of the second maximum (phimax) is always 2*pi/nfp 
    # greater than the location of the first maximum (phimin), but phimin can be either
    # 0 or pi/nfp, depending on the configuration.
    ####################################################################################
    ####################################################################################
    ####################################################################################
    if vmec.wout.bmnc[1,1] < 0:
        phimin = np.pi/nfp
    else:
        phimin = 0
    phimax = phimin + 2*np.pi/nfp

    phis2D = np.tile( np.linspace(phimin, phimax,nphi), (nalpha,1)).T

    # Sample magnetic field strengths
    Bjs = np.linspace(0,1,nBj)

    # Loop through flux snorms on which we're optimizing
    for si in range(ns):
        snorm = snorms[si]
        # Extract relevant quantities from Boozer object
        # Mode numbers
        xm_nyq = boozer.bx.xm_b
        xn_nyq = boozer.bx.xn_b

        # Fourier coefficients
        bmnc = boozer.bx.bmnc_b[:,si]

        # Rotational transform 
        iota = UnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:], k=1, s=0)(snorm)
        
        # The array that will hold the |B| values along these fieldlines
        B = np.zeros((nphi, nalpha))

        # The range of poloidal angles we're sampling
        thetamin = -iota * phimin
        thetamax = thetamin + 2*np.pi
            
        # 2D toroidal and poloidal arrays that correspond to fieldline coordinates
        thetas2D = np.tile( np.linspace(thetamin, thetamax, nalpha), (nphi,1)) + iota*phis2D

        # Loop through fourier modes, construct |B| array
        for jmn in range(len(bmnc)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * thetas2D - n * phis2D
            cosangle = np.cos(angle)
            B += bmnc[jmn] * cosangle
        
        # Normalize B between 0 and 1
        Bmin = np.min(B)
        Bmax = np.max(B)
        B = (B - Bmin) / (Bmax - Bmin)
        
        ################################################################
        ########################### SQUASH #############################
        ################################################################
        Bp_arr = np.zeros((nalpha,nphi))
        bncs = np.zeros((nalpha,nBj))
        phip_arr = np.zeros((nalpha,2*nBj-1))
        phipp_arr = np.zeros((nalpha,2*nBj-1))
        wts = np.zeros(nalpha)

        for ialpha in range(nalpha):
            # Fieldline information
            Ba = 1*B[:,ialpha]
            phisa = 1*phis2D[:,ialpha]
            # Index of the minimum of B on the fieldline
            indmin = np.argmin(Ba)

            ######## LEFT SIDE SQUASH ########
            # Define the left-hand-side of the well as everything to the left of the minimum
            Bl = 1*Ba[:indmin+1]
            phisl = phisa[:indmin+1]
            phisr = phisa[indmin:]           
            # Location of maximum of LHS of well
            indmax_l = np.argmax(Bl)
            Bl[:indmax_l] = Bl[indmax_l]
            # Squash the rest of the well
            for i in range(len(Bl)-1):
                if Bl[i] <= Bl[i+1]:
                    jf = len(Bl)-1
                    for j in range(i+1,len(Bl)):
                        if Bl[j] < Bl[i]:
                            jf = j
                            break
                    Bl[i:jf] = Bl[i]

            ######## RIGHT SIDE SQUASH ########
            # Same process for right-hand-side
            Br = 1*Ba[indmin:]
            indmax_r = np.argmax(Br)
            # If the maximum is NOT at the end of the well
            Br[indmax_r:] = Br[indmax_r]
            # Squash the rest of the well
            for j in range(len(Br)-1,1,-1):
                if Br[j-1] >= Br[j]:
                    kf = 0
                    for k in range(j-1,1,-1):
                        if Br[k] < Br[j]:
                            kf = k
                            break
                    Br[kf+1:j] = Br[j]
            
            ################################################################
            ########################### STRETCH ############################
            ################################################################
            pmax=50
            pmin=15
            def F_l():
                R1 = (1 - Bl[0])
                R2 = -Bl[-1]
                x1 = (phisl - phisl[0]) / (phisl[-1] - phisl[0])
                x1lp5 = x1 < 0.5
                t1 = x1lp5 * R1*((np.cos(2*np.pi*x1) + 1) / 2)**pmax
                t2 = (1 - x1lp5) * R2*((np.cos(2*np.pi*x1) + 1) / 2)**pmin
                return t1 + t2
            def F_r():
                 R1 = 1 - Br[-1]
                 R2 = -Br[0]
                 x1 = (phisr - phisr[0]) / (phisr[-1] - phisr[0])
                 x1lp5 = x1 < 0.5
                 t1 = x1lp5 * R2*((np.cos(2*np.pi*x1) + 1) / 2)**pmin
                 t2 = (1 - x1lp5) * R1*((np.cos(2*np.pi*x1) + 1) / 2)**pmax
                 return t1 + t2
            F_l = F_l()
            F_r = F_r()
            Bl = Bl + 1*F_l
            Br = Br + 1*F_r
            Bl = Bl[:-1]

            # The new (phi,B) fieldline
            Blr = 1*np.concatenate((Bl,Br))

            # Weights for measuring bounce distances
            wtf = UnivariateSpline(phis2D[:,ialpha], np.abs(Ba - Blr)**2)
            wts[ialpha] = (phimax - phimin) / wtf.integral(phimin,phimax)
            
            # Store magnetic field strengths
            Bp_arr[ialpha,:] = 1*Blr
            
            # Find all bounce distances of the squashed+stretched well, as well
            # as the location of the bounces
            for j in range(nBj):
                Bj = Bjs[j]
                phip1,phip2,m1,m2 = GetBranches(phisa,Blr,Bj,1,0)
                bncs[ialpha,j] = 1*(phip2-phip1)

                phip_arr[ialpha, nBj - j - 1] = 1*phip1
                phip_arr[ialpha, nBj + j - 1] = 1*phip2

        ################################################################
        ########################### SHUFFLE ############################
        ################################################################
        # Normalize weights, and calculate the weighted mean bounce
        # distances
        wts = wts / np.sum(wts)
        mbncs = (np.sum(bncs * wts[:,np.newaxis],axis=0))

        # Shuffle the well
        mean_denom = 0
        Bpps = np.concatenate( (np.flip(Bjs), Bjs[1:]) )
        for ialpha in range(nalpha):
            dbncs = (bncs[ialpha,:] - mbncs)/2
            phils = 1*phip_arr[ialpha,:nBj]
            phirs = 1*phip_arr[ialpha,nBj-1:]

            phils = phils + np.flip(dbncs)
            phirs = phirs - dbncs

            phipp_arr[ialpha,:nBj] = phils
            phipp_arr[ialpha,nBj-1:] = phirs

            Bf = UnivariateSpline(phis2D[:,ialpha],B[:,ialpha],k=1,s=0)
            try:
                Bppf = UnivariateSpline(phipp_arr[ialpha,:],Bpps,k=1,s=0)
            except:
                phils = 1*phipp_arr[ialpha,:nBj]
                phirs = 1*phipp_arr[ialpha,nBj-1:]

                for il in range(nBj-1):
                    if phils[il+1] - phils[il] < 0:
                        phirs[-il-2] = phirs[-il-2] + (phils[il] - phils[il+1] + 1e-12)
                        phils[il+1] = phils[il] + 1e-12
                    if phirs[-il-1] - phirs[-il-2] < 0:
                        phils[il+1] = phils[il+1] + (phirs[-il-1] - phirs[-il-2] - 1e-12)
                        phirs[-il-2] = phirs[-il-1] - 1e-12
                
                phipp_arr[ialpha,:nBj] = phils
                phipp_arr[ialpha,nBj-1:] = phirs

                Bppf = UnivariateSpline(phipp_arr[ialpha,:],Bpps,k=1,s=0)

            phis = np.linspace(phimin,phimax,nphi_out)

            # Penalize differences between original and shuffled wells
            denom = 1
            pen = (Bppf(phis) - Bf(phis)) / denom

            mean_denom += np.mean(denom) / nalpha

            if arr_out == True:
                out[si, ialpha, :] = np.sqrt(weights[si]) * pen / np.sqrt(nphi_out)
            else:
                out[si, ialpha] = np.sqrt(weights[si]) * np.sqrt(np.mean(pen**2))
        if arr_out == True:
            out[si,:,:] = out[si,:,:] * mean_denom
        else:
            out[si,:] = out[si,:] * mean_denom
        s = '%.2f'%(snorm)
        out = out
        # print("("+s+") =", np.sum((out[si,:] / np.sqrt(nalpha) )**2))
    out = out.flatten()
    out = out / np.sqrt(nalpha)
    return out    

# A helper function that outputs the values of phiBs when Ba=Bj
def GetBranches(phiBs,Ba,Bj,Bmax,Bmin):
    diffs = Ba - Bj
            
    # The only case when this is negative is when B = Bj somewhere between two indices
    diffsgn = diffs[0:-1]*diffs[1:]

    # Indices where Bj crosses B
    inds = np.where(diffsgn<0)[0]
    inds = np.sort(inds)

    # ind1 = first crossing, ind2 = second crossing
    # If we're at the maximum or minimum
    if Bj == Bmin or Bj < Bmin:
        imin = np.argmin(Ba)
        phimin = phiBs[imin]
        return phimin,phimin,imin,imin
    elif Bmax == Bj or Bj > Bmax:
        return phiBs[0], phiBs[-1], 0, len(Ba)-1
    
    # If there are <2 crossings, then we have found a case when Bj == Ba[ind],
     # and hence diffsign[ind] = 0, rather than being <0
    if len(inds) < 2:
        inds = np.where(diffsgn<=0)[0]
        for iind in range(1,len(inds)):
            if inds[iind] != inds[iind-1]+1:
                inds = [inds[iind-1],inds[-1]]
                break
    
    # If there are >2 crossings, we've managed to land on a flattened bit.
    # In this case, take the first and last occurances.
    if len(inds) > 2:
        inds = [inds[0],inds[-1]]
    ind1 = inds[0]
    ind2 = inds[1]

    # Linearly interpolate to find first crossing
    dy1 = Ba[ind1] - Ba[ind1+1]
    dx1 = phiBs[ind1] - phiBs[ind1+1]
    m1 = dy1/dx1
    b1 = Ba[ind1] - m1*phiBs[ind1]
    if m1 != 0:
        phiB1 = (Bj - b1)/m1
    else:
        phiB1 = phiBs[ind1]

    # Linearly interpolate to find second crossing
    dy2 = Ba[ind2] - Ba[ind2+1]
    dx2 = phiBs[ind2] - phiBs[ind2+1]
    m2 = dy2/dx2
    b2 = Ba[ind2] - m2*phiBs[ind2]
    if m2 != 0:
        phiB2 = (Bj - b2)/m2
    else:
        phiB2 = phiBs[ind2+1]

    return phiB1, phiB2, m1, m2

# Penalize the configuration's mirror ratio
#   If mirror_ratio > t, then the penalty is triggered, else penalty is zero.
# For this reason, if you're using this penalty, I suggest choosing a very
#   large weight, as this will essentially act as a "wall" and prevent the 
#   mirror ratio from exceeding whatever you set as "t"
def MirrorRatioPen(vmec,t=0.21):
    vmec.run()
    xm_nyq = vmec.wout.xm_nyq
    xn_nyq = vmec.wout.xn_nyq
    bmnc = vmec.wout.bmnc.T
    bmns = 0*bmnc
    nfp = vmec.wout.nfp
    
    Ntheta = 300
    Nphi = 300
    thetas = np.linspace(0,2*np.pi,Ntheta)
    phis = np.linspace(0,2*np.pi/nfp,Nphi)
    phis2D,thetas2D=np.meshgrid(phis,thetas)
    b = np.zeros([Ntheta,Nphi])
    for imode in range(len(xn_nyq)):
        angles = xm_nyq[imode]*thetas2D - xn_nyq[imode]*phis2D
        b += bmnc[1,imode]*np.cos(angles) + bmns[1,imode]*np.sin(angles)
    Bmax = np.max(b)
    Bmin = np.min(b)
    m = (Bmax-Bmin)/(Bmax+Bmin)
    # print("Mirror =",m)
    # pen = np.max([0,m-t])
    # return pen
    return m

# Penalize the configuration's VMEC aspect ratio
#   If aspect_ratio > t, then the penalty is triggered, else penalty is zero.
# For this reason, if you're using this penalty, I suggest choosing a very
#   large weight, as this will essentially act as a "wall" and prevent the 
#   aspect ratio from exceeding whatever you set as "t"
def AspectRatioPen(vmec,t=10):
    vmec.run()
    asp = vmec.wout.aspect
    print("Aspect Ratio =",asp)
    pen = np.max([0,asp-t])
    return pen

# Penalizes the configuration's maximum elongation
def MaxElongationPen(vmec,t=6.0,ntheta=14,nphi=6):
    """
    Penalizes the configuration's maximum elongation (e_max) if it exceeds some threshold (t).
    Specifically, if e_max > t, then output (e_max - t). Else, output zero.
    vmec        -   VMEC object
    t           -   Mximum elongation above which the output is nonzero
    ntheta      -   Number of points per poloidal cross-section
    nphi        -   Number of poloidal cross-sections
    """
    nfp = vmec.wout.nfp
    # Load variables from VMEC
    if 1 == 1:
        xm = vmec.wout.xm
        xn = vmec.wout.xn
        rmnc = vmec.wout.rmnc.T
        zmns = vmec.wout.zmns.T
        lasym = vmec.wout.lasym
        raxis_cc = vmec.wout.raxis_cc
        zaxis_cs = vmec.wout.zaxis_cs
        if lasym == True:
            raxis_cs = vmec.wout.raxis_cs
            zaxis_cc = vmec.wout.zaxis_cc
            rmns = vmec.wout.rmns
            zmnc = vmec.wout.zmnc
        else:
            raxis_cs = 0*raxis_cc
            zaxis_cc = 0*zaxis_cs
            rmns = rmnc*0
            zmnc = zmns*0

        # Set up variables
        theta1D = np.linspace(0,2*np.pi,num=ntheta)
        phi1D = np.linspace(0,2*np.pi/nfp,num=nphi)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    # A function that will return the cartesian coordinates of the boundary for a given pair of VMEC angles
    def FindBoundary(theta,phi):
        phi = phi[0]
        rb = np.sum(rmnc[-1,:] * np.cos(xm*theta + xn*phi))
        zb = np.sum(zmns[-1,:] * np.sin(xm*theta + xn*phi))
        xb = rb * np.cos(phi)
        yb = rb * np.sin(phi)

        return np.array([xb,yb,zb])

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    # Set up axis
    if 1 == 1:
        Rax = np.zeros(nphi)
        Zax = np.zeros(nphi)
        Raxp = np.zeros(nphi)
        Zaxp = np.zeros(nphi)
        Raxpp = np.zeros(nphi)
        Zaxpp = np.zeros(nphi)
        Raxppp = np.zeros(nphi)
        Zaxppp = np.zeros(nphi)
        for jn in range(len(raxis_cc)):
            n = jn * nfp
            sinangle = np.sin(n * phi1D)
            cosangle = np.cos(n * phi1D)

            Rax += raxis_cc[jn] * cosangle
            Zax += zaxis_cs[jn] * sinangle
            Raxp += raxis_cc[jn] * (-n * sinangle)
            Zaxp += zaxis_cs[jn] * (n * cosangle)
            Raxpp += raxis_cc[jn] * (-n * n * cosangle)
            Zaxpp += zaxis_cs[jn] * (-n * n * sinangle)
            Raxppp += raxis_cc[jn] * (n * n * n * sinangle)
            Zaxppp += zaxis_cs[jn] * (-n * n * n * cosangle)

            if lasym == True:
                Rax += raxis_cs[jn] * sinangle
                Zax += zaxis_cc[jn] * cosangle + zaxis_cs[jn] * sinangle
                Raxp += raxis_cs[jn] * (n * cosangle)
                Zaxp += zaxis_cc[jn] * (-n * sinangle)
                Raxpp += raxis_cs[jn] * (-n * n * sinangle)
                Zaxpp += zaxis_cc[jn] * (-n * n * cosangle)
                Raxppp += raxis_cs[jn] * (-n * n * n * cosangle)
                Zaxppp += zaxis_cc[jn] * (n * n * n * sinangle) 

        Xax = Rax * np.cos(phi1D)
        Yax = Rax * np.sin(phi1D)

        #############################################################################################
        #############################################################################################
        #############################################################################################
        #############################################################################################

        d_l_d_phi = np.sqrt(Rax * Rax + Raxp * Raxp + Zaxp * Zaxp)
        d2_l_d_phi2 = (Rax * Raxp + Raxp * Raxpp + Zaxp * Zaxpp) / d_l_d_phi

        d_r_d_phi_cylindrical = np.array([Raxp, Rax, Zaxp]).transpose()
        d2_r_d_phi2_cylindrical = np.array([Raxpp - Rax, 2 * Raxp, Zaxpp]).transpose()

        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        tangent_cylindrical = np.zeros((nphi, 3))
        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        tangent_R   = tangent_cylindrical[:,0]
        tangent_phi = tangent_cylindrical[:,1]

        tangent_Z   = tangent_cylindrical[:,2]
        tangent_X   = tangent_R * np.cos(phi1D) - tangent_phi * np.sin(phi1D)
        tangent_Y   = tangent_R * np.sin(phi1D) + tangent_phi * np.cos(phi1D)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    # Arrays that will store cross-section locations, for various poloidal angles at a fixed toroidal angle
    Xp = np.zeros(ntheta)
    Yp = np.zeros(ntheta)
    Zp = np.zeros(ntheta)
    # An array that will store the elongations at various toroidal angles
    elongs = np.zeros(nphi)

    # Loop through toroidal angles, finding the elongation of each one, and storing it in elongs
    for iphi in range(nphi):
        phi = phi1D[iphi]

        # x,y,z components of the axis tangent
        tx = tangent_X[iphi]
        ty = tangent_Y[iphi]
        tz = tangent_Z[iphi]
        t_ = np.array([tx,ty,tz])
        # x,y,z location of the axis
        xax = Xax[iphi]
        yax = Yax[iphi]
        zax = Zax[iphi]
        pax = np.array([xax, yax, zax])
        
        # Loop through poloidal angles, keeping toroidal angle fixed
        for ipt in range(ntheta):
            theta = theta1D[ipt]
            # This function returns zero when the point on the boundary is perpendicular to the axis' tangent vector
            fdot = lambda p : np.dot( t_ , (FindBoundary(theta, p) - pax) )
            # Find the cross-section's  point'
            phi_x = fsolve(fdot, phi)
            sbound = FindBoundary(theta, phi_x)
            # Subtract any noise
            sbound -= np.dot(sbound,t_)
            
            # Store cross-section locations
            Xp[ipt] = sbound[0]
            Yp[ipt] = sbound[1]
            Zp[ipt] = sbound[2]
        # Find the perimeter and area the boundary cross-section
        perim = np.sum(np.sqrt((Xp-np.roll(Xp,1))**2 + (Yp-np.roll(Yp,1))**2 + (Zp-np.roll(Zp,1))**2))
        A = FindArea(Xp,Yp,Zp)

        # Area of ellipse = A = pi*a*b
        #   a = semi-major, b = semi-minor
        # b = A / (pi*a)
        # Eccentricity = e = 1 - b**2/a**2
        #                  = 1 - A**2 / (pi**2 * a**4)
        #                  = 1 - (A / (pi * a**2))**2
        # Circumference = C = 4 * a * ellipe(e) --> Use this to solve for semi-major radius a
        #
        # b = A / (pi * a)
        # Elongation = E = semi-major / semi-minor 
        #                = a / b
        #                = a * (pi * a) / A
        #                = pi * a**2 / A
        
        # Fit an ellipse to this cross-section shape
        perim_resid = lambda a : perim - (4*a*ellipe(1 - ( A / (np.pi * a**2 ) )**2))
        if iphi == 0:
            a1 = fsolve(perim_resid, 1)
        else:
            a1 = fsolve(perim_resid, a1)
        a2 = A / (np.pi * a1)
        if a1 > a2:
            maj = a1
            min = a2
        else:
            maj = a2
            min = a1
        # Store the effective elongation
        elongs[iphi] = maj/min

    # Penalize maximum elongation
    # print("Max Elongation =",np.max(elongs))
    # print("Mean Elongation =",np.mean(elongs))
    # e = np.max(elongs)
    # pen = np.max([0,e-t])
    # return pen
    # return elongs/len(elongs)
    return elongs

# Finds unit normal vector of plane defined by points a, b, and c
# Helper function for FindArea
def FindUnitNormal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return (x/magnitude, y/magnitude, z/magnitude)

# Finds the area of polygon defined by points X, Y, and Z
# Helper function for MaxElongationPen
def FindArea(X,Y,Z):
    total = [0,0,0]

    for i in range(len(X)):
        x1 = X[i]
        y1 = Y[i]
        z1 = Z[i]
        
        x2 = X[(i+1)%(len(X))]
        y2 = Y[(i+1)%(len(Y))]
        z2 = Z[(i+1)%(len(Z))]

        vi1 = [x1,y1,z1]
        vi2 = [x2,y2,z2]

        prod = np.cross(vi1,vi2)
        total += prod
    pt0 = [X[0], Y[0], Z[0]]
    pt1 = [X[1], Y[1], Z[1]]
    pt2 = [X[2], Y[2], Z[2]]
    result = np.dot(total,FindUnitNormal(pt0,pt1,pt2))
    return abs(result/2)
