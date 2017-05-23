#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:44:02 2017

@author: max
"""

import numpy as np
from numpy import sin, cos, arcsin
from scipy import pi
from scipy.constants import mu_0, elementary_charge, k
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from math import atan

np.seterr(divide='ignore', invalid='ignore')

####################################################################
# SPECIFY PLASMA PARAMETERS
####################################################################

#BASIC PLASMA GEOMETRY 149468_1905
a = 0.6 #MINOR RADIUS
B_phi_0 = 6 # Tesla at the magnetic axis
R0_a = 1.67 #MAJOR RADIUS AT GEOMETRIC AXIS (MAGNETIC AXIS IS FUNCTION OF r DUE TO SHAFRANOV SHIFT)
Z0 = 0
kappa_up = 1.65 #1.65 #ELONGATION AT THE SEPERATRIX (upper half)
kappa_lo = 1.35 #1.3 #ELONGATION AT THE SEPERATRIX (lower half)
tri_up = 0.25 #0.2 #TRIANGULARITY AT THE SEPERATRIX (upper half)
tri_lo = 0.25 #0.1 #TRIANGULARITY AT THE SEPERATRIX (lower half)
xpt = [1.45339,-1.20574] #xpt[0] is R position, xpt[1] is Z position
theta0 = -(atan((R0_a-xpt[0])/(Z0-xpt[1]))+pi/2) #this is the angle in radians between the geometric axis and the xpoint.

#Parameters to control the meshing in the plasma region
thetapts = 50 # this number is used for both the miller model calculations and the mesh
rpts = 10 #number of radial points for 0<=rho<=1 used in miller
rmeshnum_p = 6
rmeshnum_s = 5

#PARAMETERS TO SYNTHESIZE RADIAL DENSITY, TEMPERATURE, AND CURRENT DISTRIBUTIONS
ni0 = 3.629E19
ni9 = 1.523E19
ni_sep = 0.3E19
ni_halo = 1E16
ni_dp = 1E17
nu_ni = 2.5

ne0 = 3.629E19
ne9 = 1.523E19
ne_sep = 0.3E19
ne_halo = 1E16
ne_dp = 1E17
nu_ne = 2.5

Ti0 = 8.54
Ti9 = 1.56
Ti_sep = 0.3
Ti_halo = 0.003
Ti_dp = 0.03
nu_Ti = 2.5

Te0 = 3.394
Te9 = 1.158
Te_sep = 0.06
Te_halo = 0.006
Te_dp = 0.01
nu_Te = 2.5

j0 = 1.5E6
nu_j = 1.5

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def shafranov_shift(R0_a,a,R,Z,r,theta,thetapts,tri,kappa,p,j_r_ave,s,rpts):
    
    # THIS CALCULATES A MATRIX OF THE LENGTHS OF EACH SECTION OF EACH FLUX
    # SURFACE AND THEN SUMS THEM TO GET THE PERIMETER IN 2D OF EACH FLUX
    # SURFACE (VALUE OF r).
    L_seg = np.sqrt((Z-np.roll(Z,-1,axis=1))**2 + (R-np.roll(R,-1,axis=1))**2)
    L_seg [:,-1] = 0        
    L_r = np.tile(np.sum(L_seg, axis=1), (thetapts, 1)).T
    
    #Calculate cross-sectional area for each r and corresponding differential area
    area = np.zeros(s)
    for i in range(0,rpts):
        area[i,:] = PolyArea(R[i,:],Z[i,:])
    
    diff_area = area - np.roll(area,1,axis=0)
    diff_area[0,:]=0
    
    diff_vol = diff_area * 2*pi*R0_a #approx because it uses R0_a instead of shifted R0
    vol = np.cumsum(diff_vol,axis=0)
    
    #Calculate each differential I and sum to get cumulative I
    diff_I = diff_area * j_r_ave
    I = np.cumsum(diff_I, axis=0)
    #IP = I[-1,0]
    
    #Calculate B_p_bar
    B_p_bar = mu_0 * I / L_r
    B_p_bar[0,:]=0
    
    #Calculate li
    li = (np.cumsum(B_p_bar**2 * diff_vol, axis=0) / vol) / (2*B_p_bar**2)
    li[0,:]=0
    
    #Calculate beta_p
    beta_p = 2*mu_0*(np.cumsum(p*diff_vol,axis=0)/vol-p) / B_p_bar**2
    
    #Calculate dR0dr
    dR0dr = np.zeros(s)
    R0 = np.zeros(s)
    
    f = 2*(kappa**2+1)/(3*kappa**2+1)*(beta_p+li/2)+1/2*(kappa**2-1)/(3*kappa**2+1)
    f[0,:] = f[1,:] ############ NEED TO REVISIT, SHOULD EXTRAPOLATE SOMEHOW
    
    dR0dr[-1,:] = -a*f[-1,:]/R0_a
    R0[-1,:] = R0_a
    
    for i in range(rpts-2,-1,-1):
        R0[i,:] = dR0dr[i+1,:] * (r[i,:]-r[i+1,:]) + R0[i+1,:]
        dR0dr[i,:] = -r[i,:]*f[i,:]/R0[i,:]
    
    return [R0,dR0dr,B_p_bar,L_seg]
    
def xmil(n1, n2, r, theta, a, theta0):
    
    def x_stretch(n1,n2,theta0,yscale,epsilon,thetaW):
        def f_sep(n1,n2,x):
            f = np.piecewise(
                                x,
                                [x <= -1, #condition 1
                                 np.logical_and((-1 < x),(x <= 0)), #condition 2
                                 np.logical_and(( 0 < x),(x <= 1)), #condition 3
                                 x>1], #condition 4
                                 [lambda x: 0, #function for condition 1
                                  lambda x: (1.0-abs(x)**n2)**n1, #function for condition 2
                                  lambda x: (1.0-abs(x)**n2)**n1, #function for condition 3
                                  lambda x: 0] #function for condition 4
                                 )
            return f 
    
        #xnum=201 # number of points on the x axis for the mollification. Should be odd and any integer >~ 20 will work. We're using 201 and it works well.
        if epsilon > 0: #for everywhere except the seperatrix, mollify seperatrix function with bump function
            xnum =  int(round(17/epsilon))
            x = np.linspace(-1,1,num=xnum) #num should be odd number so there is a middle point to line up exactly with theta0
            
            #Seperatrix function that will get convolved with the bump function
            f = f_sep(n1,n2,x*(1-epsilon))
            
            #Bump function that will get convolved with the seperatrix function
            eta_epsilon = np.where(abs(x)<epsilon,1/epsilon*1/0.49*np.exp(-1/(1-((x)/epsilon)**2)),0)
            
            #convolve bump and seperatrix function
            y = np.convolve(f,eta_epsilon,'same')/(xnum/2)
        else:
            xnum = 101
            x = np.linspace(-1,1,num=xnum)
            y = f_sep(n1,n2,x*(1-epsilon))
            
        #Scale the resulting function vertically to match the x-point and horizontally to meet the inboard and outboard midplanes
        y = y*yscale
        
        x = x*thetaW+theta0
        func1 = interp1d(x,y,bounds_error=0,fill_value=0)
        
        x = x + 2*pi
        func2 = interp1d(x,y,bounds_error=0,fill_value=0)
    
        return [func1,func2]
    
    xkappa1 = np.zeros(r.shape)
    xkappa2 = np.zeros(r.shape)
    for i in range(0,rpts):
        epsilon = 1-(r[i,0]/a)**1
        yscale = (r[i,0]/a)**5
        thetascale1 = 1*abs(theta0)
        thetascale2 = pi-abs(theta0)
        func1 = x_stretch(n1,n2,theta0,yscale,epsilon,thetascale1)[0] #centers function at theta0
        func2 = x_stretch(n1,n2,theta0,yscale,epsilon,thetascale2)[1] #centers function at theta0 + 2*pi
        for j in range(0,theta.shape[1]):
            xkappa1[i,j] = func1(theta[i,j])
            xkappa2[i,j] = func2(theta[i,j])  
        xkappa = xkappa1 + xkappa2    
    xkappa = yscale*xkappa*(xpt[1]/sin(theta0)/a-kappa)
    return xkappa
    


####################################################################
# MILLER MODEL CALCULATIONS
####################################################################
#In order to ensure that we have a mesh line exactly at the inboard and outboard midplanes,
#I'm defining the theta meshes in each of the four quadrants separately and then joining them
#I'll calculate about how much of the poloidal space is between each of the four specified poloidal
#locations so the meshes are approximately the same poloidal width all the way around.
#This necessarily means that the specified "thetapts" is only an approximation. The actual number
#may be slightly different.

thetapts1 = int(abs(theta0-1)/(2*pi)*thetapts) #number of poloidal mesh lines in plasma region 1 (between theta0 and outboard midplane)
thetapts2 = int(abs(pi/2-0)/(2*pi)*thetapts) #number of poloidal mesh lines in plasma region 2 (between outboard midplane and theta=pi/2)
thetapts3 = int(abs(pi-pi/2)/(2*pi)*thetapts) #number of poloidal mesh lines in plasma region 3 (between theta=pi/2 and inboard midplane)
thetapts4 = int(abs(theta0+2*pi-pi)/(2*pi)*thetapts) #number of poloidal mesh lines in plasma region 4 (between inboard midplane and theta0)
thetapts_tot = thetapts1 + thetapts2 + thetapts3 + thetapts4

#overwrite given thetapts with new one
thetapts = thetapts_tot-3
s = (rpts,thetapts) #s is the size of the matrix, used later

theta1      = np.delete(np.linspace(theta0,0,thetapts1), -1)
#soltheta1   = np.delete(np.linspace(xtheta3,0,thetapts1), -1) #angles of poloidal meshes in SOL region 1 (between xtheta3 and outboard midplane)
theta2      = np.delete(np.linspace(0,pi/2,thetapts2), -1)
theta3      = np.delete(np.linspace(pi/2,pi,thetapts3), -1)     
#note, soltheta2 and soltheta3 are calculated together (soltheta23), but
#can't be calculated until after the plasma_mesh is called because we 
#want the angles regions 2 and 3 to line up with the plasma region meshes.
#If we interpolated between inboard and outboard midplanes, it would technically
#probably work, but line coming out of pi/2 would be vertical and the adjoining
#line in the plasma region would not be vertical (assuming a plasma with nonzero triangularity)
#To deal with this, miller_func will pass back soltheta1, soltheta4, and the necessary information
#to calculate soltheta23, i.e., thetapts1 and thetapts4, because that's how many points it will need to
#delete from the beginning and end of the array it uses. See plasma_mesh for more information.    

theta4      = np.linspace(pi,theta0+2*pi,thetapts4)
#soltheta4   = np.linspace(pi,xtheta2,thetapts4) #angles of poloidal meshes in SOL region 4 (between Inboard midplane and xtheta2)

theta_tot = np.hstack((theta1,theta2,theta3,theta4))

theta = np.ones(s) * theta_tot #np.linspace(theta0,2*pi+theta0,num=thetapts)    

#CREATE THE r, theta matrices
r = np.transpose(np.transpose(np.ones(s)) * np.linspace(0,a,num=rpts))   

########################################################################
# CREATE THE KAPPA AND TRI MATRICES AND RELATED PARAMETERS
########################################################################

s_k_up = 0.1 #ELONGATION MILLER PARAMETER (upper half)
s_k_lo = 0.1 #ELONGATION MILLER PARAMETER (lower half)
   
kappa = np.where(((theta>=0)&(theta<pi)) | ((theta>=2*pi)&(theta<3*pi)), kappa_up / (a**s_k_up) * r**s_k_up,kappa_lo / (a**s_k_lo) * r**s_k_lo)
s_k   = np.where(((theta>=0)&(theta<pi)) | ((theta>=2*pi)&(theta<3*pi)), s_k_up, s_k_lo)    
tri   = np.where(((theta>=0)&(theta<pi)) | ((theta>=2*pi)&(theta<3*pi)), tri_up * r/a,tri_lo * r/a)
s_tri = np.where(((theta>=0)&(theta<pi)) | ((theta>=2*pi)&(theta<3*pi)), r*tri_up/(a*np.sqrt(1-tri)), r*tri_lo/(a*np.sqrt(1-tri)))

#X-miller extension
xmil_on=1 #turn x-point stuff on or off. It's much faster with it off.
if xmil_on == 1:
    n1 = 2
    n2 = 1
    xkappa = xmil(n1, n2, r, theta, a, theta0)
    kappa = kappa + xkappa

#synthesize density and temperature distributions based on parabola-to-a-power-on-a-pedestel model
ni     = np.where(r<0.9*a,(ni0-ni9)*(1-(r/a)**2)**nu_ni + ni9,(ni_sep-ni9)/(0.1*a)*(r-0.9*a)+ni9)
ne     = np.where(r<0.9*a,(ne0-ne9)*(1-(r/a)**2)**nu_ne + ne9,(ne_sep-ne9)/(0.1*a)*(r-0.9*a)+ne9)
Ti_kev = np.where(r<0.9*a,(Ti0-Ti9)*(1-(r/a)**2)**nu_Ti + Ti9,(Ti_sep-Ti9)/(0.1*a)*(r-0.9*a)+Ti9)
Te_kev = np.where(r<0.9*a,(Te0-Te9)*(1-(r/a)**2)**nu_Te + Te9,(Te_sep-Te9)/(0.1*a)*(r-0.9*a)+Te9)

Ti_J = Ti_kev * 1000 * elementary_charge
Te_J = Te_kev * 1000 * elementary_charge

Ti_K = Ti_kev * 1.159E7
Te_K = Te_kev * 1.159E7

#synthesize pressure profile
p = ni * k * Ti_K #this is pressure in pascals    

#synthesize current distribution, calculate each differential I and sum to get cumulative I
j_r = j0*(1-(r/a)**2)**nu_j
j_r_ave = np.roll((j_r + np.roll(j_r,-1, axis=0))/2,1,axis=0)
j_r_ave[0,:]=0

#At this point, we can calculate the flux surfaces with no shafranov shift
#This is necessary to get estimates of L_r to use for other quantities
R0 = np.ones(s) * R0_a 

R = R0 + r * cos(theta+arcsin(tri)*sin(theta))
Z = kappa*r*sin(theta)

#Update R0 and dR0dr by calling the "shafranov shift" function
shaf = shafranov_shift(R0_a,a,R,Z,r,theta,thetapts,tri,kappa,p,j_r_ave,s,rpts)
R0, dR0dr, B_p_bar, L_seg = shaf

#NOW USE UPDATED R0 AND dR0dr to get new R,Z. This can be interated again if necessary.
# I'll probably make an interation loop at some point
R = R0 + r * cos(theta+arcsin(tri)*sin(theta))
Z = kappa*r*sin(theta)

dkappa_dtheta   = np.gradient(kappa, edge_order=2)[1] * thetapts/(2*pi)
dkappa_dr       = np.gradient(kappa, edge_order=2)[0] * rpts/a

dkappa_dtheta[-1] = dkappa_dtheta[-2]
dkappa_dr[-1] = dkappa_dr[-2]
#s_k = r*

dZ_dtheta       = r*(kappa*np.cos(theta)+dkappa_dtheta*np.sin(theta))
dZ_dr           = np.sin(theta)*(r*dkappa_dr + kappa)
dR_dr           = dR0dr - np.sin(theta + np.sin(theta)*np.arcsin(tri))*(np.sin(theta)*s_tri) + np.cos(theta+np.sin(theta)*np.arcsin(tri))
dR_dtheta       = -r*np.sin(theta+np.sin(theta)*np.arcsin(tri))*(1+np.cos(theta)*np.arcsin(tri))

abs_grad_r = np.sqrt(dZ_dtheta**2 + dR_dtheta**2) / np.abs(dR_dr*dZ_dtheta - dR_dtheta*dZ_dr)

# NOW THAT WE'VE GOT R and Z PRETTY CLOSE, WE WANT TO CALCULATE THE POLOIDAL FIELD STRENGTH EVERYWHERE
# THE PROBLEM IS THAT WE'VE GOT 2 EQUATIONS IN 3 UNKNOWNS. HOWEVER, IF WE ASSUME THAT THE POLOIDAL
# INTEGRAL OF THE FLUX SURFACE AVERAGE OF THE POLOIDAL MAGNETIC FIELD IS APPROX. THE SAME AS THE
# POLOIDAL INTEGRAL OF THE ACTUAL POLOIDAL MAGNETIC FIELD, THEN WE CAN CALCULATE THE Q PROFILE
B_phi = B_phi_0 * R[0,0] / R

#Calculate initial crappy guess on q
q = B_phi_0*R[0,0] / (2*pi*B_p_bar) * np.tile(np.sum(L_seg/R**2,axis=1), (thetapts, 1)).T #Equation 16 in the miller paper. The last term is how I'm doing a flux surface average
q[0,:]=q[1,:]

dPsidr = (B_phi_0 * R[0,0]) / (2*pi*q)*np.tile(np.sum(L_seg/(R*abs_grad_r),axis=1), (thetapts, 1)).T

#f1 = (sin(theta + arcsin(tri)*sin(theta))**2 *(1+arcsin(tri)*cos(theta))**2 + kappa**2*cos(theta)**2)**(1/2) / kappa
#f2 = cos(arcsin(tri)*sin(theta))+dR0dr*cos(theta)+(s_k -s_tri*cos(theta)+(1+s_k)*arcsin(tri)*cos(theta))*sin(theta)*sin(theta+arcsin(tri)*sin(theta))

Psi = np.zeros(r.shape)
for index,row in enumerate(r):
    if index >= 1:
        Psi[index] = dPsidr[index]*(r[index,0]-r[index-1,0]) + Psi[index-1]
Psi_norm = Psi / Psi[-1,0]

B_p = dPsidr * 1/R * abs_grad_r
B_p[0,:] = 0


########################################################################
# PLOT RESULTS
########################################################################
fig = plt.figure(1,figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.axis('equal')
ax1.set_xlim(np.amin(R)*1.1,np.amax(R)*1.1)
ax1.set_ylim(np.amin(Z)*1.1,np.amax(Z)*1.1)
CS = ax1.contourf(R,Z,B_p,500) #plot something calculated by miller
plt.colorbar(CS)
#ax1.plot(R.flatten(),Z.flatten(),lw=1,color='black')
