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
print("Using S1 mask in native space")

ins=[]
indiscore=[]
pvalue=[]
cvlist=[]
for sub in range(1,sub_number+1):

    clf = LinearCSVMC(probability=True,enable_ca=['probabilities'])
    fselector = FixedNElementTailSelector(60, tail='upper',mode='select', sort=False) #select 60 voxels for MVPA
    sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,enable_ca=['sensitivities'])
    fsclf = FeatureSelectionClassifier(clf, sbfs)
    permutator=AttributePermutator('targets',count=1000) #permutation test
    null_dis_est=MCNullDist(permutator,tail='left',enable_ca=['dist_samples'])
    cv = CrossValidation(fsclf, NFoldPartitioner(attr='session'),null_dist=null_dis_est,postproc=mean_sample(),enable_ca=['stats'])


    maskname="/Users/js94538/motor/ff00%i/ref/indiS1maskth35.nii.gz" %sub #S1 mask in each subject's native space

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
inmeth=pd.DataFrame({'method':['withinsub']*sub_number,'shuffled':['No']*sub_number,'shifted':['No']*sub_number,'mask':['S1 in individual space']*sub_number})
indimean=pd.DataFrame({'mean':ins})
indf=indf.join(inmeth)
indf=indf.join(indimean)
indf.to_csv('/Users/js94538/motoranalysis/output/newS1_%s_within.txt' %methtype)
#save withinsub classification permutation test null distribution
indilist=[list(x.null_dist.ca.dist_samples.samples[0][0]) for x in cvlist]
indinulldist=pd.DataFrame({'null_error':indilist[0]+indilist[1]+indilist[2]+indilist[3]+indilist[4]+indilist[5],'condition':'MNI-inverse within subject','subject':['sub0']*1000+['sub1']*1000+['sub2']*1000+['sub3']*1000+['sub4']*1000+['sub5']*1000})
indinulldist.to_csv('/Users/js94538/motoranalysis/output/null_dist_newS1_within.txt')


#correlation FeatureSelection,ref:haxby,2011-----------------------------------------------------
def my_sum_score(me):
    maxarray=np.zeros(shape=(sub_number-1,wholeallsub[me].nfeatures))
    count=0
    for s in range(sub_number):
      if s!=me:
          corvoxels=[max(np.corrcoef(np.concatenate((wholeallsub[me].samples[:,i].reshape(-1,1),wholeallsub[s].samples),axis=1),rowvar=False)[0][1:]) for i in range(wholeallsub[me].nfeatures)]
          maxarray[count]=corvoxels
          count+=1
    myfinalscore=np.sum(maxarray,axis=0)
    return myfinalscore

for me in range(sub_number):
    my_score=my_sum_score(me)
    print("doing %i" %me)
    mask=np.isin(my_score,np.sort(my_score)[-60:]) #use 60 voxels
    allsub[me]=allsub[me][:,mask]
    wholeallsub[me]=wholeallsub[me][:,mask]
#end of correlation FeatureSelection,ref:haxby,2011-----------------------------------------------------

bsc_hyper_results=[]
bsc_means=[]
grouppvalue=[]
groupcvlist=[]
for test_run in range(n_runs):

    clf = LinearCSVMC()
    permutator=AttributePermutator('targets',count=1000)
    null_dis_est=MCNullDist(permutator,tail='left',enable_ca=['dist_samples'])
    cvtest = CrossValidation(clf, NFoldPartitioner(attr='subject'),null_dist=null_dis_est,postproc=mean_sample(),enable_ca=['stats'])

    ds_train4hyper = [sub[sub.sa.session != test_run, :] for sub in wholeallsub]
    ds_test = [sub[sub.sa.session == test_run, :] for sub in allsub]

    hyper = Hyperalignment()
    hypmaps = hyper(ds_train4hyper)
    ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, ds_test)]
    ds_hyper = vstack(ds_hyper)
    res_cv = cvtest(ds_hyper)

    p=cvtest.ca.null_prob
    groupcvlist.append(cvtest)
    grouppvalue.append(np.asscalar(p))
    print('p value: %f' %np.asscalar(p))

    bsc_mean=np.mean(res_cv)
    bsc_means.append(bsc_mean)
    bsc_hyper_results.append(res_cv)



bsc_hyper_results1=hstack(bsc_hyper_results)
print ('Mean between subject ERROR: %f' %np.mean(bsc_hyper_results1.samples))
print('Mean between subject p value: %f' %np.mean(grouppvalue))
print("All 1000 permutation test--- %s seconds ---" %(time.time()-startime))
#save betweensub classification error data
bsc=[tuple([y[0] for y in bsc_hyper_results[i].samples]) for i in range(8)]
dflabels=['sub0','sub1','sub2','sub3','sub4','sub5']
df = pd.DataFrame.from_records(bsc, columns=dflabels)
meth=pd.DataFrame({'method':['between sub']*8,'shuffled':['No']*8,'shifted':['No']*8,'mask':['S1 in individual space']*8,'sub_number':[sub_number]*8})
means=pd.DataFrame({'mean':bsc_means})
df=df.join(meth)
df=df.join(means)
df.to_csv('/Users/js94538/motoranalysis/output/NewS1_hyperalign_between.txt' %methtype)
#save betweensub classification permutation test null distribution
grouplist=[list(x.null_dist.ca.dist_samples.samples[0][0]) for x in groupcvlist]
groupnulldist=pd.DataFrame({'null_error':grouplist[0]+grouplist[1]+grouplist[2]+grouplist[3]+grouplist[4]+grouplist[5]+grouplist[6]+grouplist[7],'condition':'hyper between subject','test run':['run0']*1000+['run1']*1000+['run2']*1000+['run3']*1000+['run4']*1000+['run5']*1000+['run6']*1000+['run7']*1000})
groupnulldist.to_csv('/Users/js94538/motoranalysis/output/null_dist_hyper_between.txt')
