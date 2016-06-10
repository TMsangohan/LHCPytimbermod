# coding: utf-8
import PyTimber_Tom, pagestore
import pandas as pd
import numpy as np
import csv,glob, datetime, collections, time, subprocess,os, itertools
import madxmodule as madx

from scipy import optimize as opt
from scipy import constants as const

from pandas import HDFStore 
from collections import namedtuple

# simdata
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import scatter_matrix

# ---------------------------------------------------------------------------
# check available diskspace
# ---------------------------------------------------------------------------
DiskUsage = namedtuple('DiskUsage','Total used free')
def disk_usage(path):
    st     = os.statvfs(path)
    free   = st.f_bavail * st.f_frsize
    total  = st.f_blocks * st.f_frsize
    used   = (st.f_blocks - st.f_bfree) * st.f_frsize
    return DiskUsage(total, used,free)


# ---------------------------------------------------------------------------
# fucntion to get a list of fills in a user defined time interval
# ---------------------------------------------------------------------------
def getfills(t1,t2):
    db     = PyTimber_Tom.LoggingDB()
    fills  = db.get('HX:FILLN',t1,t2)
    t,v    = fills['HX:FILLN']
    df     = pd.DataFrame(np.vstack((t,v)).T,columns=['UnixTime','Fill'])
    df['Timestamp'] = pd.to_datetime(df['UnixTime'],unit='s')
    df['Fill'] = df['Fill'].apply(lambda x: int(x))
    return df

# --------------------
# returns fill summary
# --------------------
def getsummary(fill):
    db     = PyTimber_Tom.LoggingDB()
    if type(fill) == int:
        try:
            summary = db.getLHCFillData(fill)
            sum2 = summary['beamModes']
            summary.pop('beamModes',None)
            summary['mode'] = summary.pop('fillNumber')
            sum2[:0] = [summary]
            dfsum = pd.DataFrame.from_dict(sum2)
            dfsum['startTime']= pd.to_datetime(dfsum['startTime'],unit='s')
            dfsum['endTime']= pd.to_datetime(dfsum['endTime'],unit='s')
            dfsum['mode'] = dfsum['mode'].apply(str)
            return dfsum
        except:
            print 'Something went wrong no fill data loaded.'
    else:
        print 'Fillnumber needs to be integer ! Try again.'

# --------------------
# convert rf bucket to pos
# --------------------
def convertSlotToPos(slots):
    pos = np.array([int((s-1)/10.)  if s != 1 else 1 for s in slots])
    return pos

# -------------------------
# def convert to unix time
# -------------------------
def converttimetounix(t):
    return time.mktime(datetime.datetime.strptime(t,"%Y-%m-%d %H:%M:%S.%f").timetuple())

# ---------------------------------------------------------
# function for adding times in YY-mm-dd HH:MM:SS.fff format
# ---------------------------------------------------------
def addtime(self,intime,deltahour):
    mytime = datetime.datetime.strptime(intime,"%Y-%m-%d %H:%M:%S.%f")
    mytime += datetime.timedelta(hours=deltahour)
    return mytime.strftime("%Y-%m-%d %H:%M:%S.%f")

# ---------------------------------------------------------
# function returns 4 dataframes, 2 for each beam
# one hor and one ver containing the names
# of the used bpm around the requested ip
# and their s postions, where the BPMs left
# of that ip have negative s values
# twiss files are generated in the function cycling to
# the requested ip.
# ---------------------------------------------------------
def getbpmip(fill,ip=1):
    # get bpm data for the fill
    # +++++++++++++++++++++++++
    bpmhor, bpmver = fill.getbpmdata()
    
    # run madx to get bpm locations 
    # +++++++++++++++++++++++++++++
    fnlhcb1twiss = madx.Twiss('LHCB1','LHCB1.tfs',
                              fileloading='''
system,"ln -fns  /afs/cern.ch/eng/lhc/optics/runII/2015/ db5";
call, file="db5/lhc_as-built.seq";
option, -echo, -warn;''',
                              IPcycle='IP'+str(ip),twisscols=['NAME','S'])
    fnlhcb2twiss = madx.Twiss('LHCB2','LHCB2.tfs',
                              fileloading='''
system,"ln -fns  /afs/cern.ch/eng/lhc/optics/runII/2015/ db5";
call, file="db5/lhcb4_as-built.seq";
option, -echo, -warn;''',
                              IPcycle='IP'+str(ip),twisscols=['NAME','S'])
    
    # getting the tfs column names to name the dataframe columns
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    colsb1 = madx.get_tfsheader(fnlhcb1twiss.lower())
    colsb2 = madx.get_tfsheader(fnlhcb2twiss.lower())
    
    # reading tfs file, skipping over the meta data
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    tfsb1 = pd.read_csv(fnlhcb1twiss.lower(),skiprows=range(47),delim_whitespace=True,names=colsb1,index_col=False)
    tfsb2 = pd.read_csv(fnlhcb2twiss.lower(),skiprows=range(47),delim_whitespace=True,names=colsb2,index_col=False)
    
   
    # extracting the length of LHC to generate negative coordinates for the BPMs left of the
    # selected ip
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    s0    = tfsb1.iloc[-1]['S']
    s1    = tfsb2.iloc[-1]['S']
    
    # selecting only the bpms from the tfs file using a mask on a dataframe
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    bpmmask1 = tfsb1.apply(lambda x: ('BPM' in x['NAME']) &                        (('R'+str(ip)+'.B1' in x['NAME']) | ('L'+str(ip)+'.B1' in x['NAME'])),axis=1)
    tfsb1 = tfsb1[bpmmask1]
    
    bpmmask2 = tfsb2.apply(lambda x: ('BPM' in x['NAME']) &                        (('R'+str(ip)+'.B2' in x['NAME']) | ('L'+str(ip)+'.B2' in x['NAME'])),axis=1)
    tfsb2 = tfsb2[bpmmask2]
    
    # reshaping coordinates of the bpm locations
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    tfsb1['S'] = tfsb1.apply(lambda x: x['S'] if ('R'+str(1)+'.B1' in x['NAME']) else x['S'] - s0,axis=1)
    tfsb2['S'] = tfsb2.apply(lambda x: x['S'] if ('R'+str(1)+'.B2' in x['NAME']) else x['S'] - s0,axis=1)
    
    # selecting only the ones used during this fill
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    bpmmask11  = tfsb1.apply(lambda x: x['NAME'] in set(bpmhor.columns),axis=1)
    bpmmask111 = tfsb1.apply(lambda x: x['NAME'] in set(bpmver.columns),axis=1)
    tfsb1hor   = tfsb1[bpmmask11]
    tfsb1ver   = tfsb1[bpmmask111]
    
    bpmmask22  = tfsb2.apply(lambda x: x['NAME'] in set(bpmhor.columns),axis=1)
    bpmmask222 = tfsb2.apply(lambda x: x['NAME'] in set(bpmver.columns),axis=1)
    tfsb2hor   = tfsb2[bpmmask22]
    tfsb2ver   = tfsb2[bpmmask222]
    
    # removing the twiss files to save space
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    os.remove('lhcb1.tfs')
    os.remove('lhcb2.tfs')
    return tfsb1hor,tfsb1ver,tfsb2hor,tfsb2ver

class LHCfill(object):
    # constants
    protonmass = const.physical_constants['proton mass energy equivalent in MeV'][0]/1000 # GeV
    ionA       = 208.
    ionZ       = 82.
    energy     = 6370
    gamma      = energy * ionZ / 193.7291748489224

    # beta's for the undulators and dipoles for the bsrt light
    betaUndH = [203.,200.]
    betaUndV = [318.,327.]
    betaDipH = [214., 205.]
    betaDipV = [328.,344.]
    
    def __init__(self,fillnumber,storefile):
        self.db            = PyTimber_Tom.LoggingDB()
        self.fillnumber    = fillnumber
        self.storefile     = storefile
        self.hdffile= HDFStore(storefile)
        
        # getting fill summary
        # ++++++++++++++++++++
        if '/summary' in self.hdffile.keys():
            self.dfsummary = self.hdffile['/summary']
        else:
            self.dfsummary     = getsummary(fillnumber)
        self.hdffile.close()
        
        self.hdffile= HDFStore(storefile)
        # getting filling scheme
        # ++++++++++++++++++++++
        self.fillingscheme = self.getFillingScheme()
        self.hdffile.close()
        
        self.hdffile= HDFStore(storefile)
        # writing to hdf5 file
        # ++++++++++++++++++++++
        self.hdffile.put('/summary',self.dfsummary,format='t')
        self.hdffile.put('/fillingscheme',self.fillingscheme,format='t')
        
        self.hdffile.close()
        self.hdffile= HDFStore(storefile)
        # getting bunch postions
        # ++++++++++++++++++++++
        if '/bunchpos' in self.hdffile.keys():
            self.dfbunchpos = self.hdffile['/bunchpos']
        else:
            data = self.getVarList(['LHC.BQM.B1:FILLED_BUCKETS', 'LHC.BQM.B2:FILLED_BUCKETS'])
            self.dfbunchpos = self.getbunchpositions(data)
         
        self.hdffile.close()
        self.hdffile= HDFStore(storefile)
        # getting bunch lengths
        # +++++++++++++++++++++
        if ('/bunchlengthb1' in self.hdffile.keys()) & ('/bunchlengthb2' in self.hdffile.keys()):
            self.dfbunchlenb1 = self.hdffile['/bunchlengthb1']
            self.dfbunchlenb2 = self.hdffile['/bunchlengthb2']
        else:
            data = self.getVarList(['LHC.BQM.B1:BUNCH_LENGTHS','LHC.BQM.B2:BUNCH_LENGTHS'])
            self.dfbunchlenb1, self.dfbunchlenb2 = self.getbunchlengths(data)
        
        self.hdffile.close()
        self.hdffile= HDFStore(storefile)
        # getting transverse emittances
        # +++++++++++++++++++++++++++++
        if ('/emit/ex1' in self.hdffile.keys()) & ('/emit/ex2' in self.hdffile.keys()) &                     ('/emit/ey1' in self.hdffile.keys()) & ('/emit/ey2' in self.hdffile.keys()):
            self.dfex1 = self.hdffile['/emit/ex1']
            self.dfey1 = self.hdffile['/emit/ey1']
            self.dfex2 = self.hdffile['/emit/ex2']
            self.dfey2 = self.hdffile['/emit/ey2']
        
        else:
            data = self.getVarList(['LHC.BSRT.5R4.B1:FIT_SIGMA_H','LHC.BSRT.5R4.B1:FIT_SIGMA_V',
                                    'LHC.BSRT.5R4.B1:GATE_DELAY','LHC.BSRT.5L4.B2:FIT_SIGMA_H',
                                    'LHC.BSRT.5L4.B2:FIT_SIGMA_V','LHC.BSRT.5L4.B2:GATE_DELAY'])
            self.dfex1, self.dfey1, self.dfex2,self.dfey2 = self.getemit(data)
        
        self.hdffile.close()
        self.hdffile= HDFStore(storefile)
        # getting the bunch intensities
        # ++++++++++++++++++++++++++++++
        if (('/fbct/b1' in self.hdffile.keys()) & ('/fbct/b2' in self.hdffile.keys())):
            self.fbctb1 = self.hdffile['/fbct/b1']
            self.fbctb2 = self.hdffile['/fbct/b2']
        else:
            data = self.getVarList(['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY','LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY'])
            self.fbctb1, self.fbctb2 = self.getfbct(data)
        
        self.hdffile.close()
        self.hdffile= HDFStore(storefile)
        # getting the instantenous luminosities
        # +++++++++++++++++++++++++++++++++++++
        if ('/lumi' in self.hdffile.keys()):
            self.dflumi = self.hdffile['/lumi']
        else:
            data = self.getVarList(["ATLAS:LUMI_TOT_INST",
                                    "ALICE:LUMI_TOT_INST",
                                    "CMS:LUMI_TOT_INST",
                                    "LHCB:LUMI_TOT_INST"])
            self.dflumi = self.getlumi(data)
            
        # closing the hdf5 file which makes it write the data in memory to the file
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hdffile.close()

    # ----------------------
    # returns filling scheme
    # ----------------------
    def getFillingScheme(self):
        if '/fillingscheme' in self.hdffile.keys():
            return self.hdffile['/fillingscheme']
        else:
            start = self.dfsummary[self.dfsummary['mode']==str(self.fillnumber)]['endTime'].values[0]
            t1= pd.Timestamp(start)
            schemes = self.db.get('LHC:INJECTION_SCHEME',t1)
            t,v =schemes['LHC:INJECTION_SCHEME']
            dfFillScheme = pd.DataFrame(np.vstack((t,v)).T,columns=['TimeStamp','Scheme'])
            dfFillScheme['Scheme'] =  dfFillScheme['Scheme'].apply(str)
            dfFillScheme['TimeStamp'] =  dfFillScheme['TimeStamp'].apply(float)
            return dfFillScheme

    # ----------------------
    # getting all the data
    # ----------------------
    def getVarList(self,variables):
        t1 = pd.Timestamp(self.dfsummary[self.dfsummary['mode']==str(self.fillnumber)]['startTime'].values[0])
        t2 = pd.Timestamp(self.dfsummary[self.dfsummary['mode']==str(self.fillnumber)]['endTime'].values[0])
        data = self.db.get(variables,t1,t2)
        return data
    
    # ---------------------------------------------------------
    # returns two column df with the bunch slots for each beam in a column called respectively 
    # b1pos and b2pos
    # ---------------------------------------------------------
    def getbunchpositions(self,data):
        t1,v1 = data['LHC.BQM.B1:FILLED_BUCKETS']
        t2,v2 = data['LHC.BQM.B2:FILLED_BUCKETS']

        # converting to integers as bunch buckets are integers after all
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        v1 = v1.astype(int)
        v2 = v2.astype(int)

        # removing all the rows with only zeros and selecting the last row
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        v1nonzero = v1[~np.all(v1==0,axis=1)][-1,:]
        v2nonzero = v2[~np.all(v2==0,axis=1)][-1,:]

        # selecting only the non zero values
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        bunchslotsb1 = v1nonzero[np.nonzero(v1nonzero)]
        bunchslotsb2 = v2nonzero[np.nonzero(v2nonzero)]


        # creating and filling dataframe
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        dfbunchpos = pd.DataFrame(convertSlotToPos(bunchslotsb1),columns=['b1pos'])
        dfbunchpos['b2pos'] =  convertSlotToPos(bunchslotsb2)

        # writing to hdf5 file
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hdffile.put('/bunchpos',dfbunchpos,format='t')
        return dfbunchpos
    
    # ---------------------------------------------------------
    # returns two dataframes containing the bunch lenghts for 
    # each beam, column names are the bunchslots of the
    # bunches in each beam respectively
    # ---------------------------------------------------------
    def getbunchlengths(self,data):
        try:
            t1,v1 = data['LHC.BQM.B1:BUNCH_LENGTHS']
            t2,v2 = data['LHC.BQM.B2:BUNCH_LENGTHS']
            
            # loading the data in a dataframes and rename the columns such that
            # the column name corresponds to the bunchslot of that bunch
            # times are in unix times
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            dfb1  =  pd.DataFrame(v1)
            dfb1  = dfb1[range(len(self.dfbunchpos['b1pos']))]
            dfb1.columns = list(self.dfbunchpos['b1pos'])
            dfb1['t'] = t1
            
            dfb2  =  pd.DataFrame(v2)
            dfb2  = dfb2[range(len(self.dfbunchpos['b2pos']))]
            dfb2.columns = list(self.dfbunchpos['b2pos'])
            dfb2['t'] = t2
            
            # writing to hdf5 file
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            self.hdffile.put('/bunchlengthb1',dfb1,format='t')
            self.hdffile.put('/bunchlengthb2',dfb2,format='t')
            return dfb1,dfb2
        except:
            print 'Something went wrong no data loaded'
            return 0
    
    # ---------------------------------------------------------
    # returns four dataframes containing the tranverse emittance
    # the column names are the the bunchslots of the
    # bunches in each beam respectively
    # ---------------------------------------------------------
    def getemit(self,data):
        try:
            # loading the data arrays
            # +++++++++++++++++++++++
            t1, gd1       = data['LHC.BSRT.5R4.B1:GATE_DELAY']
            t11, sighb1   = data['LHC.BSRT.5R4.B1:FIT_SIGMA_H']
            t111, sigvb1  = data['LHC.BSRT.5R4.B1:FIT_SIGMA_V']
            t2, gd2       = data['LHC.BSRT.5L4.B2:GATE_DELAY']
            t22, sighb2   = data['LHC.BSRT.5L4.B2:FIT_SIGMA_H']
            t222, sigvb2  = data['LHC.BSRT.5L4.B2:FIT_SIGMA_V']

            # converting to dataframes and filtering 
            # ++++++++++++++++++++++++++++++++++++++
            dfgd1         = pd.DataFrame(list(gd1))
            fgdb1conv     = dfgd1.fillna(value=0).astype(int)
            bunchesb1     = fgdb1conv.drop(0,axis=1).as_matrix()
            bunchsetb1    = set(bunchesb1.flatten())

            dfgd2         = pd.DataFrame(list(gd2))
            fgdb2conv     = dfgd2.fillna(value=0).astype(int)
            bunchesb2     = fgdb2conv.drop(0,axis=1).as_matrix()
            bunchsetb2    = set(bunchesb2.flatten())

            dfsighb1      = pd.DataFrame(list(sighb1))
            dfsighb1['t'] = t11
            dfsigvb1      = pd.DataFrame(list(sigvb1))
            dfsigvb1['t'] = t111

            dfsighb2      = pd.DataFrame(list(sighb2))
            dfsighb2['t'] = t22
            dfsigvb2      = pd.DataFrame(list(sigvb2))
            dfsigvb2['t'] = t222

            dfex1         = pd.DataFrame()
            dfey1         = pd.DataFrame()
            dfex2         = pd.DataFrame()
            dfey2         = pd.DataFrame()

            for b in bunchsetb1:
                            bunchmask      = (fgdb1conv==b)
                            hor            = dfsighb1.where(bunchmask)
                            ver            = dfsigvb1.where(bunchmask)
                            hortempdf      = pd.DataFrame()
                            hortempdf['t'] = pd.Series(dfsighb1.iloc[hor.dropna(how='all').mean(axis=1).index]['t'])
                            hortempdf[b]   = hor.dropna(how='all').mean(axis=1)
                            vertempdf      = pd.DataFrame()
                            vertempdf['t'] = pd.Series(dfsigvb1.iloc[ver.dropna(how='all').mean(axis=1).index]['t'])
                            vertempdf[b]   = ver.dropna(how='all').mean(axis=1)
                            dfex1          = dfex1.append(hortempdf)
                            dfey1          = dfey1.append(vertempdf)

            for b in bunchsetb2:
                            bunchmask      = (fgdb2conv==b)
                            hor            = dfsighb2.where(bunchmask)
                            ver            = dfsigvb2.where(bunchmask)
                            hortempdf      = pd.DataFrame()
                            hortempdf['t'] = pd.Series(dfsighb2.iloc[hor.dropna(how='all').mean(axis=1).index]['t'])
                            hortempdf[b]   = hor.dropna(how='all').mean(axis=1)
                            vertempdf      = pd.DataFrame()
                            vertempdf['t'] = pd.Series(dfsigvb2.iloc[ver.dropna(how='all').mean(axis=1).index]['t'])
                            vertempdf[b]   = ver.dropna(how='all').mean(axis=1)
                            dfex2          = dfex2.append(hortempdf)
                            dfey2          = dfey2.append(vertempdf)
           
            # get end of ramp for switching betas
            # +++++++++++++++++++++++++++++++++++
            if 'RAMP' in list(self.dfsummary['mode']):
                rampend = self.dfsummary[self.dfsummary['mode']=='RAMP']['endTime'].values[0]
                rampend = (rampend.astype('uint64')/1e9)#.astype('uint32')
            else:
                rampend = self.dfsummary[self.dfsummary['mode']==str(self.fillnumber)]['endTime'].values[0]
            
            # resetting index to timestamps to decide on which beta to use
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            dfex1set = dfex1.set_index('t')
            dfey1set = dfey1.set_index('t')
            dfex2set = dfex1.set_index('t')
            dfey2set = dfex1.set_index('t')

            dfex1set[dfex1set.index>rampend] =                 dfex1set[dfex1set.index>rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaDipH[0])
            dfex1set[dfex1set.index<=rampend] =                 dfex1set[dfex1set.index<=rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaUndH[0])
            dfey1set[dfey1set.index>rampend] =                 dfey1set[dfey1set.index>rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaDipV[1])
            dfey1set[dfey1set.index<=rampend] =                 dfey1set[dfey1set.index<=rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaUndV[1])

            dfex2set[dfex2set.index>rampend] =                 dfex2set[dfex2set.index>rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaDipH[0])
            dfex2set[dfex2set.index<=rampend] =                 dfex2set[dfex2set.index<=rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaUndH[0])
            dfey2set[dfey2set.index>rampend] =                 dfey2set[dfey2set.index>rampend].applymap(lambda x: 
                                                              self.gamma * float(x)**2/self.betaDipV[1])
            dfey2set[dfey2set.index<=rampend] =                 dfey2set[dfey2set.index<=rampend].applymap(lambda x: 
                                                                    self.gamma * float(x)**2/self.betaUndV[1])
            
            # resetting index 
            # ++++++++++++++++
            ex1dfout = dfex1set.reset_index()
            ey1dfout = dfey1set.reset_index()
            ex2dfout = dfex2set.reset_index()
            ey2dfout = dfey2set.reset_index()
    
            # writing to hdf5 file
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            self.hdffile.put('/emit/ex1',ex1dfout,format='t')
            self.hdffile.put('/emit/ey1',ey1dfout,format='t')
            self.hdffile.put('/emit/ex2',ex2dfout,format='t')
            self.hdffile.put('/emit/ey2',ey2dfout,format='t')
            
            return ex1dfout,ey1dfout,ex2dfout,ey2dfout
        except:
            print 'Something went wrong no data loaded.'
            return 0
    
    # ---------------------------------------------------------
    # returns two dataframes containing the bunch intensities for 
    # each beam, column names are the bunchslots of the
    # bunches in each beam respectively
    # ---------------------------------------------------------
    def getfbct(self,data):
        try:
            t1, i1 = data['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY']
            t2, i2 = data['LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY']
            
            # loading into dataframes
            # +++++++++++++++++++++++
            dfi1 = pd.DataFrame(i1)
            dfi1 = dfi1.astype(int)
            
            # selecting only the columns corresponding to correct bunchslots
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            b1pos = list(self.dfbunchpos['b1pos'])
            b1pos[0]=0
            dfi1 = dfi1[b1pos]
            dfi1['t'] = t1
        
            # loading into dataframes
            # +++++++++++++++++++++++
            dfi2 = pd.DataFrame(i2)
            dfi2 = dfi2.astype(int)
            
            # selecting only the columns corresponding to correct bunchslots
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            b2pos = list(self.dfbunchpos['b2pos'])
            b2pos[0]=0
            dfi2 = dfi2[b2pos]
            dfi2['t'] = t2
            
            # writing to hdf5 file
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            self.hdffile.put('/fbct/b1',dfi1,format='t')
            self.hdffile.put('/fbct/b2',dfi2,format='t')
            
            return dfi1,dfi2
        except:
            print 'Something went wrong no data loaded.'
        
    # ---------------------------------------------------------
    # returns a dataframes containing the luminosity
    # for each of the experiments
    # ---------------------------------------------------------
            
    def getlumi(self,data):
        try:
            t1, atlas = data["ATLAS:LUMI_TOT_INST"]
            t2, alice = data["ALICE:LUMI_TOT_INST"]
            t3, cms = data["CMS:LUMI_TOT_INST"]
            t4, lhcb = data["LHCB:LUMI_TOT_INST"]

            dflumiatlas = pd.DataFrame(np.transpose(np.vstack((t1,atlas))))
            dflumialice = pd.DataFrame(np.transpose(np.vstack((t2,alice))))
            dflumicms   = pd.DataFrame(np.transpose(np.vstack((t3,cms))))
            dflumilhcb  = pd.DataFrame(np.transpose(np.vstack((t4,lhcb))))

            dflumiatlas.columns = ['t','atlas']
            dflumialice.columns=['t','alice']
            dflumicms.columns = ['t','cms']
            dflumilhcb.columns=['t','lhcb']

            dflum = dflumiatlas.append(dflumialice)
            dflum = dflum.append(dflumicms)
            dflum = dflum.append(dflumilhcb) 
            dflum.sort('t')
            dflum = dflum.reset_index(drop=True)
            
            self.hdffile.put('/lumi',dflum,format='t')
            return dflum
        except:
            print 'Something went wrong no data loaded.'
            
            
    def getbpmdata(self):
        hdf = HDFStore(self.storefile)
        if ('/bpm/hor' in hdf.keys()) & ('/bpm/ver' in hdf.keys()):
            dfbpmhor = hdf['/bpm/hor']
            dfbpmver = hdf['/bpm/ver']
        else:
            varnamelist = ['LHC.BOFSU:BPM_NAMES_H','LHC.BOFSU:BPM_NAMES_V']
            varlist     = ['LHC.BOFSU:POSITIONS_H','LHC.BOFSU:POSITIONS_V']    

            db = PyTimber_Tom.LoggingDB()

            t1 = pd.Timestamp(self.dfsummary[self.dfsummary['mode']==str(self.fillnumber)]['startTime'].values[0])
            t2 = pd.Timestamp(self.dfsummary[self.dfsummary['mode']==str(self.fillnumber)]['endTime'].values[0])

            bpmnames = db.get(varnamelist,t2)

            th,vh = bpmnames[varnamelist[0]]
            tv,vv = bpmnames[varnamelist[1]]

            hcolnames = list(vh[0].astype(str))
            hcolnames.append('t')
            vcolnames = list(vv[0].astype(str))
            vcolnames.append('t')

            namelist = [hcolnames,vcolnames]

            sizelist = db.getsizeest(varlist,t1,t2)
            datalist = []
            dflist   = []

            for i in range(len(sizelist)):
                if int(sizelist[i]) > 250 :
                    ninterval    = int(sizelist[i]) / 250 + 1
                    delta        = (t2-t1)/ ninterval
                    intervallist = [[(j * delta) + t1,(j+1)*delta + t1] for j in range(ninterval)]
                    datalist.append([db.get(varlist[i],times[0],times[1]) for times in intervallist])
                else:
                    datalist.append([db.get(varlist[i],t1,t2)])

            
            for i in range(len(sizelist)):
                df = pd.DataFrame()
                for item in datalist[i]:
                    t,v = item[varlist[i]]
                    dftemp = pd.DataFrame(v)
                    dftemp['t'] = t
                    df = df.append(dftemp)
                df = df.sort_values(by='t')
                df.columns = namelist[i]
                dflist.append(df)
                
            dfbpmhor = dflist[0]
            dfbpmver = dflist[1]
            
            masks           = db.get(['LHC.BOFSU:BPM_MASK_H','LHC.BOFSU:BPM_MASK_V'],t1,t2)
            t1,v1           = masks['LHC.BOFSU:BPM_MASK_H']
            t2,v2           = masks['LHC.BOFSU:BPM_MASK_V']
            
            hmaskdf         = pd.DataFrame(v1)
            hmaskdf['t']    = t1
            hmaskdf         = (hmaskdf.astype(int) >0)
            hmaskdf.columns = dfbpmhor.columns
            dfbpmhor        = dfbpmhor[hmaskdf].dropna(axis=1,how='all')
            
            vmaskdf         = pd.DataFrame(v2)
            vmaskdf['t']    = t2
            vmaskdf         = (vmaskdf.astype(int) >0)
            vmaskdf.columns = dfbpmver.columns
            dfbpmver        = dfbpmver[vmaskdf].dropna(axis=1,how='all')
            
            hdf.put('/bpm/hor',dfbpmhor,format='fixed')
            hdf.put('/bpm/ver',dfbpmver,format='fixed')
            hdf.close()
        return dfbpmhor,dfbpmver
