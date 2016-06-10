# coding: utf-8
import pandas as pd
import csv
import numpy as np
import scipy as sc
import collections
import glob
import subprocess
import datetime

from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.patches import Rectangle

from itertools import islice
from scipy import constants as const

from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ----------------------------------------------------------------
# cern specific modules for reading tfs files - outdated by pandas
# ----------------------------------------------------------------
import cern_pymad_domain_tfs as dom
import cern_pymad_io_tfs as io
# ****************************************************************

# ----------------------------------------------------------------
# constants
# ----------------------------------------------------------------
npart            = 1600
beampipediam     = 0.022    #[m]
lhcradius        = 2803.95  #[m]
dipolelength     = 14.3     #[m]
ionmass          = 193.72917484892244  #[GeV]
ioncharge        = 82.      #[el charge]
electronmassgev  = const.electron_mass * const.c**2/const.value('electron volt-joule relationship')/10**9 #[GeV]

# ----------------------------------------------------------------
# MADX - Twiss columns as dictionary
# ----------------------------------------------------------------
MADtwissColumns = {}

MADtwissColumns["RMatrixExtended"] = ["NAME", "KEYWORD", "PARENT", 
   "S", "L", "X", "PX", "Y", "PY", "T", "PT", "BETX", "BETY", "ALFX", 
   "ALFY", "MUX", "MUY", "DX", "DY", "DPX", "DPY", "HKICK", "VKICK", 
   "K0L", "K1L", "KMAX", "KMIN", "CALIB", "RE11", "RE12", 
   "RE13", "RE14", "RE15", "RE16", "RE21", "RE22", "RE23", "RE24", 
   "RE25", "RE26", "RE31", "RE32", "RE33", "RE34", "RE35", "RE36", 
   "RE41", "RE42", "RE43", "RE44", "RE45", "RE46", "RE51", "RE52", 
   "RE53", "RE54", "RE55", "RE56", "RE61", "RE62", "RE63", "RE64", 
   "RE65", "RE66"]

MADtwissColumns["LHCTwiss"] = ["NAME", "KEYWORD", "PARENT", "S", "L", 
   "LRAD", "KICK", "HKICK", "VKICK", "ANGLE", "K0L", "K1L", "K2L", 
   "K3L", "X", "Y", "PX", "PY", "BETX", "BETY", "ALFX", "ALFY", "MUX",
    "MUY", "DX", "DY", "DPX", "DPY", "KMAX", "KMIN", "CALIB", 
   "POLARITY", "APERTYPE",'APER_1','APER_2','APER_3','APER_4',"N1", "TILT"]

MADtwissColumns["CTE"] = ["NAME","S","L","BETX","BETY","ALFX","ALFY","DX","DPX","DY","DPY","ANGL","K1L","K1S"]

# ----------------------------------------------------------------
# get column names of the tfs file
# ----------------------------------------------------------------
def get_tfsheader(tfsfile):
    headerdata =  pd.read_csv(tfsfile,delim_whitespace=True, skiprows=range(45),nrows=2,index_col=None)
    return headerdata.columns[1:].values

# ----------------------------------------------------------------
# get column names of the csv file
# ----------------------------------------------------------------
def get_csvheader(csvfile):
    headerdata =  pd.read_csv(csvfile,delim_whitespace=True,nrows=2,index_col=None)
    return headerdata.columns[0:].values

# ----------------------------------------------------------------
# MADX - Error template to include errors
# ----------------------------------------------------------------
errseqtemplate='''
USE, PERIOD=LHCB2;

EOPTION,ADD=TRUE,SEED=62971100;

SELECT, FLAG=ERROR, PATTERN="MQX*.*L5";
EALIGN, DX:=TGAUSS(1.5)*1.0E-4,DY:=TGAUSS(1.5)*1.0E-4,
        DS:=TGAUSS(1.5)*1.0E-4;
        
ERR = TGAUSS(1.0)*1.0E-4;

SELECT, FLAG=ERROR, PATTERN="MQX*.*L5";
EFCOMP, ORDER:=1, RADIUS=0.010,
    DKNR={0,ERR,ERR*1.,ERR*1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        DKSR={0,ERR,ERR*1.,ERR*1.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        
'''

# ----------------------------------------------------------------
# MADX - SIXTRACK programs files path
# ----------------------------------------------------------------
sixpath          = '/afs/cern.ch/work/t/tomerten/sixtrack/'
madx             = '/afs/cern.ch/user/m/mad/bin/madx_dev'
madfilespath     = '/afs/cern.ch/work/t/tomerten/mad/'


# ----------------------------------------------------------------
# Off momentum ions delta 
#
# usage: 
# dpPb(dm,dq)
#
# dm = change in mass
# dq = change in charge
# ----------------------------------------------------------------
def dpPb(dm,dq):
    return (1+dm/ionmass)/(1+dq/ioncharge)-1

# ----------------------------------------------------------------
# Get the momentum (PC) from a tfs file 
#
# usage: 
# get_p(tfsfile)
#
# ----------------------------------------------------------------
def get_p(tfsfile):
    opt  = io.tfsDict(tfsfile)
    return opt[1]['pc']

# ----------------------------------------------------------------
# Get the kicker names in tfs file  
#
# usage: 
# get_kickernames(tfsfile,type)
# type = 'HKICKER' 
# ----------------------------------------------------------------
def get_kickernames(fn,ktype='HKICKER'):
    dat = pd.read_csv(fn,delim_whitespace=True,skiprows=range(45),index_col=None)
    dat = dat[dat.NAME != '%s']
    datheaders= dat.columns[1:]
    dat = pd.read_csv(filen,delim_whitespace=True,header=None,names=datheaders,skiprows=range(46),index_col=False)
    dat = dat[dat.KEYWORD != '%s']
    return dat[dat.KEYWORD == ktype].NAME.values

# ----------------------------------------------------------------
# MADX - R matrices for some optical elements used to calculate 
# impact locations
# ----------------------------------------------------------------

# dipole R matrix 3 X 3
def Mdipole(s,R):
    return np.array([[np.cos(s/R),R * np.sin(s/R),R * (1-np.cos(s/R))],                     [-1/R * np.sin(s/R),np.cos(s/R),np.sin(s/R)],                     [0,0,1]])

# dipole R matrix 6 X 6
def Mdipole6D(s,R):
    return np.array([
            [np.cos(s/R),R * np.sin(s/R),0,0,0,R * (1 - np.cos(s/R))],\
            [-1/R * np.sin(s/R),np.cos(s/R),0,0,0,np.sin(s/R)],\
            [0, 0, 1, s, 0, 0],\
            [0, 0, 0, 1, 0, 0],\
            [0, 0, 0, 0, 1, 0],\
            [0, 0, 0, 0, 0, 1]])

# dipole R matrix 6 X 6 for beam 4 note the sign change for the dispersion
def Mdipole6DB4(s,R):
    return np.array([
            [np.cos(s/R),R * np.sin(s/R),0,0,0,-R * (1 - np.cos(s/R))],\
            [-1/R * np.sin(s/R),np.cos(s/R),0,0,0,-np.sin(s/R)],\
            [0, 0, 1, s, 0, 0],\
            [0, 0, 0, 1, 0, 0],\
            [0, 0, 0, 0, 1, 0],\
            [0, 0, 0, 0, 0, 1]])

# focussing quadrupole R matrix 3 X 3
def Mquadf(s,k):
    return np.array([
            [np.cos(np.sqrt(np.absolute(k) * s)),\
                      (1/np.sqrt(k))*np.sin(np.sqrt(np.absolute(k) * s))],\
            [-np.sqrt(np.absolute(k))*np.sin(np.sqrt(np.absolute(k) * s)),\
                      np.cos(np.sqrt(np.absolute(k) * s))]])

# drift space R matrix 2 X 2
def Mdrift(s):
    return np.array([[1, s],[0,1]]) 

# drift space R matrix 6 X 6
def Mdrift6D(s):
    return np.array([[1, s, 0, 0, 0, 0],                    [0, 1, 0, 0, 0, 0],                    [0, 0, 1, s, 0, 0],                    [0, 0, 0, 1, 0, 0],                    [0, 0, 0, 0, 1, 0],                    [0, 0, 0, 0, 0, 1]])

# ----------------------------------------------------------------
# MADX - get initial parameters
#
# usage:
# get_initial(fn,deltap)
# fn = tfs file
# deltap = deltap to set for secondary beams
# ----------------------------------------------------------------
def get_initial(fn,deltap,location='IP5'):
    dat = pd.read_csv(fn,delim_whitespace=True,header=None,names=MADtwissColumns["LHCTwiss"],skiprows=range(47),                      index_col=False)
    dcout = {}
    info = dat[dat.NAME == location]
    dcout['BETX'] = info.BETX.values[0]
    dcout['BETY'] = info.BETY.values[0]
    dcout['ALFX'] = info.ALFX.values[0]
    dcout['ALFY'] = info.ALFY.values[0]
    dcout['X'] = info.X.values[0]
    dcout['PX'] = info.PX.values[0]
    dcout['Y'] = info.Y.values[0]
    dcout['PY'] = info.PY.values[0]
    dcout['DELTAP'] = deltap
    return dcout

# ----------------------------------------------------------------
# MADX - adding madx code to save variables values
#
# usage:
# madxknob_save(knoblist,fn)
# fn = filename where to save the variables
# knoblist = list of variables to save  ex. ON_X2
# ----------------------------------------------------------------
def madxknob_save(knoblist,fn):
    outstr = '''
SELECT,FLAG=SAVE,CLEAR;
'''
    for name in knoblist:
        outstr += 'SELECT, FLAG=SAVE, CLASS=VARIABLE, PATTERN='+ name +';\n'
    outstr += 'SAVE,FILE={fnknob};\n'
    return outstr.format(fnknob=fn)

# ----------------------------------------------------------------
# MADX - matching of an orbit bump
# usage:
# madMatchBump(lhcseq,targetxc,targetel,correctorlist)
# lhcseq = sequence string 
# targetxc = desired displacement at target element
# targetel = element where the orbit displacement should match a certain value
# correctorlist = list of correctors to use for the construction of the bump
# ----------------------------------------------------------------
def madMatchBump(lhcseq,targetxc,targetel,correctorlist):
    linelist = []
    line = ''
    linelist.append('MATCH, SEQUENCE=' + lhcseq + line + ';')
    linelist.append('CONSTRAINT, SEQUENCE=' + lhcseq + ', RANGE=' + targetel + ', X=' + str(targetxc) + ';')
    linelist.append('CONSTRAINT, SEQUENCE=' + lhcseq + ', RANGE=' + correctorlist[-1] + ', X=0, PX=0;')
    for c in correctorlist:
        linelist.append('VARY, NAME='+c+'->HKICK, STEP=1.0E-6;')
    linelist.append('LMDIF, CALLS=500, TOLERANCE=1.0E-15;')
    linelist.append('ENDMATCH;')
    out=''
    for ln in linelist:
        out += ln + '\n' 
    return out

# ----------------------------------------------------------------
# MADX - generating madx beam commands to include in a madx input file
# usage:
# MADX_Beam(beam,seq='LHCB1',bv=1,energy=522340,charge=ioncharge,mass=ionmass,kbunch=518,npart=2.E8,\
#               ex=3.5e-6,ey=3.5e-6,et=0.000013665,sige=0.00015,sigt=0.09107)
# beam = 1 or 2 to give beam a name
# seq = sequence string 
# ----------------------------------------------------------------
def MADX_Beam(beam,seq='LHCB1',bv=1,energy=522340,charge=ioncharge,mass=ionmass,kbunch=518,npart=2.E8,              ex=3.5e-6,ey=3.5e-6,et=0.000013665,sige=0.00015,sigt=0.09107):
    gam = energy / mass
    ex  = ex /gam
    ey  = ey /gam
    beamdict={
        'beam'    : 'beam'+ str(beam),
        'seq'     : seq,
        'bv'      : bv,
        'energy'  : energy,
        'charge'  : charge,
        'mass'    : '{:18.14f}'.format(mass),
        'kbunch'  : kbunch,
        'npart'   : '{:.1E}'.format(npart),
        'ex'      : ex,
        'ey'      : ey,
        'et'      : et,
        'sige'    : sige,
        'sigt'    : sigt
    }
    return '''
{beam}:  BEAM, SEQUENCE={seq}, BV={bv}, ENERGY={energy},CHARGE={charge},
    MASS={mass}, KBUNCH={kbunch},NPART={npart},
    EX={ex}, EY={ey},ET={et},SIGE={sige},SIGT={sigt};
        
    '''.format(**beamdict)

# ----------------------------------------------------------------
# MADX - running twiss
#
# usage:
#  Twiss(seq,fn,fileloading,targetxc=0.0,targetel='',correctorlist='',errorseq='',twisscols=MADtwissColumns["LHCTwiss"]
#                        ,beam=[MADX_Beam(1,seq='LHCB1',ey=1.5e-6),MADX_Beam(2,seq='LHCB2',ey=1.5e-6)])
# seq = sequence string 
# fileloading = multiline string for calling seq files and setting optics directory cusin links
# targetxc = desired displacement at target element
# targetel = element where the orbit displacement should match a certain value
# correctorlist = list of correctors to use for the construction of the bump
# errorseq = madx code for the errors to include in the twiss calculations
# twisscols = columns to write to the twiss output tfs file
# beam = list of two beam commands to use in the twiss
# ----------------------------------------------------------------
def Twiss(seq,fn,fileloading='''
system,"ln -fns  /afs/cern.ch/eng/lhc/optics/runII/2015/ db5";
call, file="db5/lhcb4_as-built.seq";
call, file="db5/opt_inj_colltunes.madx";
call, file="db5/opt_800_800_800_3000_ion_coll.madx";'''
          , optstart='#S',optstop='#E',IPcycle='IP5',targetxc=0.0,targetel=' ',correctorlist=[],\
          errorseq='',twisscols=MADtwissColumns["LHCTwiss"],beam=[MADX_Beam(1,seq='LHCB1',ey=1.5e-6),
                                                                  MADX_Beam(2,seq='LHCB2',ey=1.5e-6)]):
    tw     = ''
    cycle  = '''
SEQEDIT, SEQUENCE={seq};
FLATTEN;
CYCLE, start={ip};
ENDEDIT;

    '''.format(ip=IPcycle,seq=seq)
    
    for i in twisscols:
        tw += i+','
    dformat ={
        'beam1'     : beam[0],
        'beam2'     : beam[1],
        'start'     : optstart,
        'stop'      : optstop,
        'cycle'     : cycle,
        'seq'       : seq,
        'fn'        : fn,
        'twcol'     : tw[:-1],
        'errors'    : errorseq,
        'fileload'  : fileloading
    }
        
    if (targetel==' '):
        dformat['bumpmatch'] = ' '
    else:
        dformat['bumpmatch'] =madMatchBump(seq,targetxc,targetel,correctorlist)
        
    madin ='''
{fileload}
        
{beam1}
{beam2}
{cycle}

USE, PERIOD={seq};

SELECT,FLAG=TWISS,CLEAR;
        
SELECT,FLAG=TWISS,RANGE=#S/#E,COLUMN={twcol};
TWISS,SEQUENCE={seq},file={fn};

{bumpmatch}

{errors}

SELECT,FLAG=TWISS,CLEAR;
        
SELECT,FLAG=TWISS,RANGE=#S/#E,COLUMN={twcol};
TWISS,SEQUENCE={seq},file={fn};

system, "rm db5";

'''.format(**dformat)
    fn = open('madin.madx','wt')
    fn.write(madin)
    fn.close()
    bashcommand = madx +' < ' + 'madin.madx' 
    subprocess.call(bashcommand,shell=True)
    return dformat['fn']

# ----------------------------------------------------------------
# MADX - transfermatrices
#
# usage:
# TransferMatrix(lhcseq,optstart,optstop,initdict,targetxc,targetel,correctorlist,errorseq='',fileext=''
#                     ,beam=[MADX_Beam(1,seq='LHCB1',ey=1.5e-6), MADX_Beam(2,seq='LHCB2',ey=1.5e-6)])
# lhcseq = sequence string 
# optstart = name of element where to start the twiss
# targetxc = desired displacement at target element
# targetel = element where the orbit displacement should match a certain value
# correctorlist = list of correctors to use for the construction of the bump
# errorseq = madx code for the errors to include in the twiss calculations
# twisscols = columns to write to the twiss output tfs file
# beam = list of two beam commands to use in the twiss
# ----------------------------------------------------------------
def TransferMatrix(lhcseq,optstart,optstop,initdict,targetxc,targetel,correctorlist,
                   fileloading='''
system,"ln -fns  /afs/cern.ch/eng/lhc/optics/runII/2015/ db5";
call, file="db5/lhcb4_as-built.seq";
call, file="db5/opt_inj_colltunes.madx";
call, file="db5/opt_800_800_800_3000_ion_coll.madx";''',
                   IPcycle='IP5',\
                   errorseq='',fileext='',beam=[MADX_Beam(1,seq='LHCB1',ey=3.5e-6),
                                                                  MADX_Beam(2,seq='LHCB2',ey=3.5e-6)]):
    fnstem = "BFPPbeamTransferMatrix" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    moufn  = fnstem + '\BFPPbeamTransferMatrix.mou'
    tw     = ''
    cycle  = '''
SEQEDIT, SEQUENCE={seq};
FLATTEN;
CYCLE, start={ip};
ENDEDIT;

    '''.format(ip=IPcycle,seq=lhcseq)
        
    for i in MADtwissColumns["RMatrixExtended"]:
        tw += i+','
    
    line = ''
    for k in initdict.keys():
        line += ','+ str(k)+ '=' + str(initdict[k])
        
    
    d = {        'beam1'        : beam[0],
        'beam2'        : beam[1],
        'seq'          : lhcseq,
        'start'        : optstart,
        'stop'         : optstop,
        'cycle'        : cycle,
        'twcol'        : tw[:-1],
        'transfertfsfn': 'bfppbeamtransfermatrix' + fileext 
             + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.tfs',
        'initdc'       : line,
        'bumpmatch'    : madMatchBump(lhcseq,targetxc,targetel,correctorlist),
        'fnknob'       : 'knob'
             + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.str',
        'errors'       : errorseq,
        'fileload'     : fileloading        
        }
    madin = '''
{fileload}
       
{beam1}
{beam2}
{cycle}

USE, PERIOD={seq};
        
SELECT,FLAG=TWISS,CLEAR;
        
SELECT,FLAG=TWISS,RANGE={start}/{stop},COLUMN={twcol};
TWISS,SEQUENCE={seq},file={transfertfsfn}{initdc}, RMATRIX=TRUE;

{bumpmatch}

{errors}

SELECT,FLAG=TWISS,CLEAR;
        
SELECT,FLAG=TWISS,RANGE={start}/{stop},COLUMN={twcol};
TWISS,SEQUENCE={seq},file={transfertfsfn}{initdc}, RMATRIX=TRUE;

system, "rm db5";

    '''.format(**d)
    fn = open('madin.madx','wt')
    fn.write(madin)
    fn.close()
    bashcommand = madx +' < ' + 'madin.madx' 
    subprocess.call(bashcommand,shell=True)
    return d['transfertfsfn']

# ----------------------------------------------------------------
# MADX - Tracking of sigma matrix and generation of distributions 6D
#
# usage:
# TrackSigmaMatrix(infile,nameprevel,initdc,nparticles)
# infile = tfs input file
# nameprevel = name of element where to get the initial sigma matrix, at start of element
# initdc = initial conditions
# nparticles = number of particles in the distributions
# ----------------------------------------------------------------
def TrackSigmaMatrix(infile,nameprevel,initdc,nparticles):
    transfermatrixopt  = io.tfsDict(infile)
    prevelinex         = transfermatrixopt[0]["name"].index(nameprevel)-1
    gamma              = transfermatrixopt[1]['gamma']
    ex                 = transfermatrixopt[1]['ex']
    ey                 = transfermatrixopt[1]['ey']
    sige               = transfermatrixopt[1]['sige']
    sigt               = transfermatrixopt[1]['sigt']
    print transfermatrixopt[0]['name'][prevelinex-1]
    print transfermatrixopt[0]['name'][prevelinex]
    print transfermatrixopt[0]['name'][prevelinex+1]
    
    matrix = np.array([
        [transfermatrixopt[0]["re11"][prevelinex],transfermatrixopt[0]["re12"][prevelinex],
         transfermatrixopt[0]["re13"][prevelinex],transfermatrixopt[0]["re14"][prevelinex],
         transfermatrixopt[0]["re15"][prevelinex],transfermatrixopt[0]["re16"][prevelinex]],
        [transfermatrixopt[0]["re21"][prevelinex],transfermatrixopt[0]["re22"][prevelinex],
         transfermatrixopt[0]["re23"][prevelinex],transfermatrixopt[0]["re24"][prevelinex],
         transfermatrixopt[0]["re25"][prevelinex],transfermatrixopt[0]["re26"][prevelinex]],
        [transfermatrixopt[0]["re31"][prevelinex],transfermatrixopt[0]["re32"][prevelinex],
         transfermatrixopt[0]["re33"][prevelinex],transfermatrixopt[0]["re34"][prevelinex],
         transfermatrixopt[0]["re35"][prevelinex],transfermatrixopt[0]["re36"][prevelinex]],
        [transfermatrixopt[0]["re41"][prevelinex],transfermatrixopt[0]["re42"][prevelinex],
         transfermatrixopt[0]["re43"][prevelinex],transfermatrixopt[0]["re44"][prevelinex],
         transfermatrixopt[0]["re45"][prevelinex],transfermatrixopt[0]["re46"][prevelinex]],
        [transfermatrixopt[0]["re51"][prevelinex],transfermatrixopt[0]["re52"][prevelinex],
         transfermatrixopt[0]["re53"][prevelinex],transfermatrixopt[0]["re54"][prevelinex],
         transfermatrixopt[0]["re55"][prevelinex],transfermatrixopt[0]["re56"][prevelinex]],
        [transfermatrixopt[0]["re61"][prevelinex],transfermatrixopt[0]["re62"][prevelinex],
         transfermatrixopt[0]["re63"][prevelinex],transfermatrixopt[0]["re64"][prevelinex],
         transfermatrixopt[0]["re65"][prevelinex],transfermatrixopt[0]["re66"][prevelinex]]
    ])
    
    sigma0matrix = np.array([
        ex / 2. * np.array([initdc['BETX'],-initdc['ALFX'],0.,0.,0.,0.]),
        ex / 2. * np.array([-initdc['ALFX'],(2+initdc['ALFX']**2)/initdc['BETX'],0.,0.,0.,0.]),
        ey / 2. * np.array([0.,0.,initdc['BETY'],-initdc['ALFY'],0.,0.]),
        ey / 2. * np.array([0.,0.,-initdc['ALFY'],(2+initdc['ALFY']**2)/initdc['BETY'],0,0]),
        np.array([0.,0.,0.,0.,0.,0.]),
        np.array([0.,0.,0.,0.,0.,sige**2]),
        
    ])
    
    orbit = np.array([transfermatrixopt[0]["x"][prevelinex],
                     transfermatrixopt[0]["px"][prevelinex],
                     transfermatrixopt[0]["y"][prevelinex],
                     transfermatrixopt[0]["py"][prevelinex],
                     transfermatrixopt[0]["t"][prevelinex],
                     transfermatrixopt[0]["pt"][prevelinex]])
            
    sigma1matrix = np.dot(matrix,np.dot(sigma0matrix,np.transpose(matrix)))
    # print sigma0matrix,initdc['DELTAP'],np.sqrt(sigma1matrix[5,5])
    coordinates = np.transpose(
        np.array([
                np.random.normal(orbit[0],np.sqrt(sigma1matrix[0,0]),nparticles),
                np.random.normal(orbit[1],np.sqrt(sigma1matrix[1,1]),nparticles),
                np.random.normal(orbit[2],np.sqrt(sigma1matrix[2,2]),nparticles),
                np.random.normal(orbit[3],np.sqrt(sigma1matrix[3,3]),nparticles),
                np.random.normal(0,sigt,nparticles),
                np.random.normal(initdc['DELTAP'],np.sqrt(sigma1matrix[5,5]),nparticles)#initdc['DELTAP']
                
                         ])
    )
    return coordinates

# ----------------------------------------------------------------
# MADX - calculating impact coordinates
#
# usage:
# impactcoordinates6D(coord,ax,r,s0,ldipole,delta,mass,deltam,p0,filen,beam4=False
# coord = output of tracksigmatrix, input distributions to track
# ax = diameter of beam pipe assumed circular
# r = bending radius of main lhc dipoles
# s0 = s value of start of element where to track
# delta = output of dpPb function
# mass = particle mass
# deltam = mass change of the particles
# p0 = reference momentum
# filename = filename where to save the impact distributions
# beam4 = if tracking with beam 4 in madx set to true
# ----------------------------------------------------------------
def impactcoordinates6D(coord,ax,r,s0,ldipole,delta,mass,deltam,p0,filen,beam4=False):
    def f(s,axs,rr,y,vect):
        if beam4:
            out = -np.sqrt(axs**2-y**2)-np.dot(Mdipole6DB4(s,rr),vect)[0]
        else:
            out = np.sqrt(axs**2-y**2)-np.dot(Mdipole6D(s,rr),vect)[0]
        return out

    
    def fd(s,axs,rr,y,vect):
        if beam4:
            out = -np.sqrt(axs**2-y**2)-(np.dot(Mdrift6D(s),np.dot(Mdipole6DB4(ldipole,rr),vect)))[0]
        else:
            out = np.sqrt(axs**2-y**2)-(np.dot(Mdrift6D(s),np.dot(Mdipole6D(ldipole,rr),vect)))[0]
        return out
    
    spos = []
    for i in range(len(coord)):
        try:
            zero = brentq(f,0,100,args=(ax,r,coord[i,2],coord[i]))
            if zero > ldipole:
#                 plt.plot(range(0,100,1),[fd(j,ax,r,coord[i,2],coord[i]) for j in range(0,100,1)])
                spos.append(ldipole + brentq(fd,0,100,args=(ax,r,coord[i,2],coord[i])))
            else:
                spos.append(zero)
        except:
            continue
    #plt.show()
   
    # matrix6d = np.array([np.dot(Mdipole6D(spos[i],r),coord[i]) for i in range(len(coord))])
    if beam4:
        matrix6d = np.array([np.dot(Mdrift6D(spos[i]-ldipole),np.dot(Mdipole6DB4(ldipole,r),coord[i]))
            if spos[i] > ldipole else np.dot(Mdipole6DB4(spos[i],r),coord[i]) 
            for i in range(len(spos))])
    else:
        matrix6d = np.array([np.dot(Mdrift6D(spos[i]-ldipole),np.dot(Mdipole6D(ldipole,r),coord[i]))
                if spos[i] > ldipole else np.dot(Mdipole6D(spos[i],r),coord[i]) 
                for i in range(len(spos))])
#     print spos
    scol  = pd.Series(data=spos)+s0
    xcol  = pd.Series(data=matrix6d[:,0])
    pxcol = pd.Series(data=matrix6d[:,1])
    ycol  = pd.Series(data=matrix6d[:,2])
    pycol = pd.Series(data=matrix6d[:,3])
    Ecol  = pd.Series(data=
                      np.sqrt((mass + deltam)**2 + (p0 * (1 + matrix6d[:,5] - delta) * (1 + deltam/mass))**2))             
    
    outdf             = pd.DataFrame(scol,columns=['s[m]'])
    outdf['x[mm]']    = pd.Series(data=xcol)*1000
    outdf['px[1e-3]'] = pd.Series(data=pxcol)*1000
    outdf['y[mm]']    = pd.Series(data=ycol)*1000
    outdf['py[1e-3]'] = pd.Series(data=pycol)*1000
    outdf['E[GeV]']   = pd.Series(data=Ecol) 
    outdf.to_csv(filen + '.csv',index=False)
    return outdf
