from __future__ import division
import numpy as np
from mvpa2.suite import *
from scipy.signal import lfilter
import time
startime=time.time()
allsub=[] #keep 'taskds', for classifier training
wholeallsub=[] #keep 'wholedataset', for hyperalighment
n_runs=8

methtype='6subs'
sub_number=6
print("Using S1 mask in MNI space")

ins=[]
indiscore=[]
pvalue=[]
cvlist=[]
for sub in range(1,sub_number+1):

    clf = LinearCSVMC(probability=True,enable_ca=['probabilities'])
    fselector = FixedNElementTailSelector(60, tail='upper',mode='select', sort=False) #select 60 voxels for MVPA
    sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,enable_ca=['sensitivities'])
    # create classifier with automatic feature selection
    fsclf = FeatureSelectionClassifier(clf, sbfs)

    permutator=AttributePermutator('targets',count=1000) #permutation test
    null_dis_est=MCNullDist(permutator,tail='left',enable_ca=['dist_samples'])
    cv = CrossValidation(fsclf, NFoldPartitioner(attr='session'),null_dist=null_dis_est,postproc=mean_sample(),enable_ca=['stats'])
    #errorfx=lambda p, t: np.mean(p == t)

    maskname="/Users/js94538/motor/ff001/ref/jueLS1th35.nii.gz" %sub #S1 mask in MNI space

    import pandas as pd
    beha=pd.read_csv("/Users/js94538/motor/ff00%i/ref/ft-data-sess1.txt" %sub)
    behavior=beha[::10]

    dataset=[]
    for i in range(1,9):
        ds=fmri_dataset("/Users/js94538/motor/ff00%i/bold/sess1/rrun-00%i.nii" %(sub,i),mask=maskname)
        ds.sa['session']=[i-1]*ds.nsamples
        dataset.append(ds)
    allds=vstack(dataset,a=0)

    #detrend, Z-scoring
    detrender=PolyDetrendMapper(polyord=1,chunks_attr="session")
    deallds=allds.get_mapped(detrender)
    zscore(deallds,chunks_attr='session')
    wholedataset=deallds.copy(deep=True,sa=['session']) #dataset for hyperalighment
    filtera = np.array([1./3,1./3,1./3])
    filterb = np.array(1.)
    #moving average per 3 TRs for the clssifier training set
    for i in range(8):
        now=deallds.samples[i*180:i*180+180]
        nowl=lfilter(filtera, filterb, now, axis=0)
        deallds.samples[i*180:i*180+180]=nowl

    #select finger-pressing TRs for clssifier training
    a=([False]*21+([True]*5+[False]*3)*19+[True]*5+[False]*2)*8
    taskds=deallds[a]

    #get the pressing finger for each TR
    condtr=beha[::2]
    condtrp=condtr['probe'].values
    temparray=np.array([9]*1440) #9 means this TR doesn't belong to any finger,ie. ITI
    temparray[a]=condtrp
    wholedataset.sa['press']=temparray


    taskds.sa['session']=condtr['run_num'].values
    val=condtr['probe'].values
    taskds.sa['targets']=val
    taskds.sa['subject']=[sub]*taskds.nsamples
    wholedataset.sa['subject']=[sub]*wholedataset.nsamples


    #select the last TR in each pressing period
    #ie. the average of last 3 finger-pressing TRs in each trial(due to moving average)
    last3trs=([False]*4+[True])*160
    taskds=taskds[last3trs]

    wholeallsub.append(wholedataset[:,]) #keep 'wholedataset', for hyperalighment
    allsub.append(taskds) #keep 'taskds', for classifier training
    cv_results=cv(taskds)
    p=cv.ca.null_prob
    cvlist.append(cv)
    ins.append(np.mean(cv_results)) #individual score average
    pvalue.append(np.asscalar(p))
    indiscore.append(cv_results) #individual score for 6 left out runs
    print(np.asscalar(p))
meanins=np.mean(ins)
meanp=np.mean(pvalue)
print("Mean individual error is %f" %meanins)
print("Mean individual p_value is %f" %meanp)
print("individual 1000 permutation test--- %s seconds ---" %(time.time()-startime))

#save withinsub classification error data
within=[tuple([y[0] for y in indiscore[i].samples]) for i in range(sub_number)]
indflabels=['run0','run1','run2','run3','run4','run5','run6','run7']
indf = pd.DataFrame.from_records(within, columns=indflabels)
inmeth=pd.DataFrame({'method':['withinsub']*sub_number,'shuffled':['No']*sub_number,'shifted':['No']*sub_number,'mask':['MNI S1']*sub_number})
indimean=pd.DataFrame({'mean':ins})
indf=indf.join(inmeth)
indf=indf.join(indimean)
indf.to_csv('/Users/js94538/motoranalysis/output/MNIS1_%s_within.txt' %methtype)
#save withinsub classification permutation test null distribution
indilist=[list(x.null_dist.ca.dist_samples.samples[0][0]) for x in cvlist]
indinulldist=pd.DataFrame({'null_error':indilist[0]+indilist[1]+indilist[2]+indilist[3]+indilist[4]+indilist[5],'condition':'MNI within subject','subject':['sub0']*1000+['sub1']*1000+['sub2']*1000+['sub3']*1000+['sub4']*1000+['sub5']*1000})
indinulldist.to_csv('/Users/js94538/motoranalysis/output/null_dist_MNI_within.txt')


#between subject anatomical alignment classification
clf = LinearCSVMC()
fselector = FixedNElementTailSelector(60, tail='upper',mode='select', sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,enable_ca=['sensitivities'])
fsclf = FeatureSelectionClassifier(clf, sbfs)
permutator=AttributePermutator('targets',count=1000)
null_dis_est=MCNullDist(permutator,tail='left',enable_ca=['dist_samples'])

allsub=vstack(allsub)
cvgroup = CrossValidation(fsclf, NFoldPartitioner(attr='subject'),null_dist=null_dis_est,postproc=mean_sample(),enable_ca=['stats'])
anatomical_btw_sub=cvgroup(allsub)
p=cvgroup.ca.null_prob
print("Anatomical alignment Across subject error is %f" %np.mean(anatomical_btw_sub))
print("Anatomical alignment Across subject p_value is %f" %np.asscalar(p))
print("All 1000 permutation test--- %s seconds ---" %(time.time()-startime))

#save anatomical BSC permutation test null distribution
groupnulldis=cvgroup.null_dist.ca.dist_samples
grouplist=list(groupnulldis.samples[0][0])
df=pd.DataFrame({'null_error':grouplist,'conditoon':'MNI between subject'})
df.to_csv('/Users/js94538/motoranalysis/output/null_dist_MNI_between.txt')

#save anatomical BSC error data
lis=[x[0] for x in anatomical_btw_sub.samples]
df=pd.DataFrame(data={'mean':lis,'method':'anatolical_alignment_btwsub','mask':'MNI S1'})
df.to_csv('/Users/js94538/motoranalysis/output/MNIS1_anatomical_btw.txt')
