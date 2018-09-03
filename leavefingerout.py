from __future__ import division
import numpy as np
from mvpa2.suite import *
from scipy.signal import lfilter
allsub=[] #keep 'taskds', for classifier training
wholeallsub=[] #keep 'wholedataset', for hyperalighment
n_runs=8

methtype='6subs'
AveragedTR4hyper='No'
sub_number=6
n_fingers=2 #Number of fingers used for hyperalighment, change this line.
fingerforhyper='012' #change this line AND a line below to select Hyperalignment finger

clf = LinearCSVMC()
cv = CrossValidation(clf, NFoldPartitioner(attr='session'),errorfx=lambda p, t: np.mean(p == t))

ins=[]
indiscore=[]
for sub in range(1,sub_number+1):
    maskname="/Users/js94538/motor/ff00%i/ref/mask_lh_s1.nii" %sub

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
    wholedataset=deallds.copy(deep=True,sa=['session'])
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


    condtr=beha[::2]
    condtrp=condtr['probe'].values
    temparray=np.array([9]*1440) #9 means this TR doesn't belong to any finger,ie. ITI
    temparray[a]=condtrp
    wholedataset.sa['press']=temparray

    #For select hyperalighment fingers
    fingermask=np.isin(wholedataset.sa['press'],[0,1,2])########### change this line to select Hyperalignment finger ##################
    wholedataset=wholedataset[fingermask]

    only3tr=([False]*2+[True]*3)*120
    wholedataset=wholedataset[only3tr]

    taskds.sa['session']=condtr['run_num'].values
    val=condtr['probe'].values
    taskds.sa['targets']=val
    taskds.sa['subject']=[sub]*taskds.nsamples
    wholedataset.sa['subject']=[sub]*wholedataset.nsamples

    #Last TR in each pressing period
    last3trs=([False]*4+[True])*160
    taskds=taskds[last3trs]

    wholeallsub.append(wholedataset[:,])
    allsub.append(taskds)


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
    #my_scores.append(my_score)
    mask=np.isin(my_score,np.sort(my_score)[-60:]) #use 60 voxels
    allsub[me]=allsub[me][:,mask]
    wholeallsub[me]=wholeallsub[me][:,mask]

#end of correlation FeatureSelection,ref:haxby,2011-----------------------------------------------------------

#between subjects classification
from itertools import compress
storer={'finger0':[],'finger1':[],'finger2':[],'finger3':[],'overall':[]}
# permuaccuracy={'permuf0':[],'permuf1':[],'permuf2':[],'permuf3':[],'overall':[]}
permut={'permuf0':[],'permuf1':[],'permuf2':[],'permuf3':[],'overall':[]}
for test_run in range(n_runs):
    ds_train4hyper = [sub[sub.sa.session != test_run, :] for sub in wholeallsub]
    ds_test = [sub[sub.sa.session == test_run, :] for sub in allsub]

    hyper = Hyperalignment()
    hypmaps = hyper(ds_train4hyper)
    ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, ds_test)]
    ds_hyper = vstack(ds_hyper)
    clf = LinearCSVMC()
    #Classification
    print('for real')
    for test_sub in range(1,sub_number+1):
        test_hyper=ds_hyper[np.isin(ds_hyper.sa['subject'],[test_sub])]
        train_hyper=ds_hyper[~np.isin(ds_hyper.sa['subject'],[test_sub])]
        print(train_hyper.sa['targets'][:10])
        clf.train(train_hyper)
        prediction=clf.predict(test_hyper.samples)

        for test_finger in range(4):
            test_hyper_now=test_hyper[test_hyper.sa.targets==test_finger]
            prediction_now=list(compress(prediction,test_hyper.sa.targets==test_finger))
            storer['finger%i' %test_finger].append(np.mean(prediction_now==test_hyper_now.sa.targets))
            storer['overall'].append(np.mean(prediction_now==test_hyper_now.sa.targets))
        clf.untrain()
    #permutation test
    #classification with shuffled targets
    print('for random')
    for permu in range(1000):
        permuaccuracy={'permuf0':[],'permuf1':[],'permuf2':[],'permuf3':[],'overall':[]}
        ds_hyper_perm=ds_hyper.copy(deep=True)
        for test_sub in range(1,sub_number+1):
            test_hyper=ds_hyper[np.isin(ds_hyper_perm.sa['subject'],[test_sub])]
            train_hyper=ds_hyper[~np.isin(ds_hyper_perm.sa['subject'],[test_sub])]
            randomvalue=train_hyper.sa['targets'].value
            np.random.shuffle(randomvalue)
            train_hyper.sa['targets']=randomvalue
            print(train_hyper.sa['targets'][:10])
            clf.train(train_hyper)
            prediction=clf.predict(test_hyper.samples)

            for test_finger in range(4):
                test_hyper_now=test_hyper[test_hyper.sa.targets==test_finger]
                prediction_now=list(compress(prediction,test_hyper.sa.targets==test_finger))
                permuaccuracy['permuf%i' %test_finger].append(np.mean(prediction_now==test_hyper_now.sa.targets))
                permuaccuracy['overall'].append(np.mean(prediction_now==test_hyper_now.sa.targets))
            clf.untrain()
        for f in range(4):
            permut['permuf%i'%f].append(np.mean(permuaccuracy['permuf%i' %f]))
        permut['overall'].append(np.mean(permuaccuracy['overall']))

allpermuacc=[]
permulationover=list(np.mean(np.array(permut['overall']).reshape(8,1000),axis=0))
print(np.mean(permulationover))
for f in range(4):
    eachpermu=np.array(permut['permuf%i'%f])
    eachpermu=eachpermu.reshape(8,1000)

    allpermuacc+=list(np.mean(eachpermu,axis=0))
    fac=np.mean(storer['finger%i' %f])
    p=sum([fac<x for x in list(np.mean(eachpermu,axis=0))]) / float(1000)
    print('p value for finger %i is %f' %(f,p) )



# save permutation test null distribution
dfper=pd.DataFrame(data={'score':allpermuacc+permulationover,'test_finger':['finger0']*1000+['finger1']*1000+['finger2']*1000+['finger3']*1000+['overall']*1000,'finger used for hyperalign':'finger%s' %fingerforhyper})
dfper.to_csv('/Users/js94538/motoranalysis/output/permu_leave_finger_%s_out.txt' %fingerforhyper)
# save classification accuracy for each finger
df=pd.DataFrame(data={'score':storer['finger0']+storer['finger1']+storer['finger2']+storer['finger3']+storer['overall'],'test_finger':['finger0']*48+['finger1']*48+['finger2']*48+['finger3']*48+['overall']*192,'finger used for hyperalign':'finger%s' %fingerforhyper})
df.to_csv('/Users/js94538/motoranalysis/output/leave_finger_out_%s.txt' %fingerforhyper)
