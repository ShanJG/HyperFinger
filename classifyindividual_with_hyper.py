from __future__ import division
import numpy as np
from mvpa2.suite import *
from scipy.signal import lfilter

methtype='6subs'
AveragedTR4hyper='No'
sub_number=6

#change following 2 lines as needed
indihyper=3 #number of runs to train hyoeralignemnt, 3 maybe max out.
trainnumber=12 #number of runs to train classifier
#there are 16 runs, leave at least 1 run for testing

clf = LinearCSVMC()

allpossirun=list(itertools.combinations(range(8*2),indihyper))
np.random.shuffle(allpossirun)
restrun=[list(set(range(8*2)).difference(set(list(x)))) for x in allpossirun]
map(np.random.shuffle,restrun)
storer={'sub0':[],'sub1':[],'sub2':[],'sub3':[],'sub4':[],'sub5':[]}
hyperstorer={'sub0':[],'sub1':[],'sub2':[],'sub3':[],'sub4':[],'sub5':[]}

for hugeiteration in range(8): #randomly select runs for hyperalign,training and testing, for 8 times.
    allsub=[] #keep 'taskds', for classifier training
    allsubtest=[] #keep 'taskds_test', for classifier testing
    wholeallsub=[] #keep 'wholedataset', for hyperalighment training

    #following is index of runs for hyperalighment training, classifier training and classifier testing
    run2hyper=list(allpossirun[hugeiteration])
    print('run to hyper:')
    print(run2hyper)
    run2train=restrun[hugeiteration][0:trainnumber]
    print('run to train:')
    print(run2train)
    run2test=list(restrun[hugeiteration][trainnumber:])
    print("run to test:")
    print(run2test)

    for sub in range(1,sub_number+1):
        maskname="/Users/js94538/motor/ff00%i/ref/mask_lh_s1.nii" %sub

        import pandas as pd
        behasession1=pd.read_csv("/Users/js94538/motor/ff00%i/ref/ft-data-sess1.txt" %sub)
        behasession2=pd.read_csv("/Users/js94538/motor/ff00%i/ref/ft-data-sess2.txt" %sub)
        behasession2['run_num']=behasession2['run_num']+8
        beha=pd.concat([behasession1,behasession2],axis=0,ignore_index=True)
        behavior=beha[::10]

        dataset=[]
        for i in range(1,9):
            ds=fmri_dataset("/Users/js94538/motor/ff00%i/bold/sess1/rrun-00%i.nii" %(sub,i),mask=maskname)
            ds.sa['session']=[i-1]*ds.nsamples
            dataset.append(ds)
        for i in range(1,9):
            ds=fmri_dataset("/Users/js94538/motor/ff00%i/bold/sess2/rrun-00%i.nii" %(sub,i),mask=maskname)
            ds.sa['session']=[i-1+8]*ds.nsamples
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
        for i in range(8*2):
            now=deallds.samples[i*180:i*180+180]
            nowl=lfilter(filtera, filterb, now, axis=0)
            deallds.samples[i*180:i*180+180]=nowl


        #select finger-pressing TRs for clssifier training
        a=([False]*21+([True]*5+[False]*3)*19+[True]*5+[False]*2)*8*2
        taskds=deallds[a]

        #get the pressing finger for each TR
        condtr=beha[::2]
        condtrp=condtr['probe'].values
        temparray=np.array([9]*1440*2) #9 means this TR doesn't belong to any finger,ie. ITI
        temparray[a]=condtrp
        wholedataset.sa['press']=temparray
        wholedataset.sa['subject']=[sub]*wholedataset.nsamples

        taskds.sa['session']=condtr['run_num'].values
        val=condtr['probe'].values
        taskds.sa['targets']=val
        taskds.sa['subject']=[sub]*taskds.nsamples

        #Last TR in each pressing period
        last3trs=([False]*4+[True])*160*2
        taskds=taskds[last3trs]

        runmask=np.isin(wholedataset.sa['session'],run2hyper)
        runmask1=np.isin(taskds.sa['session'],run2train)
        runmask2=np.isin(taskds.sa['session'],run2test)
        wholedataset=wholedataset[runmask] #hyperalighment training dataset
        taskds_train=taskds[runmask1] #classifier training dataset
        taskds_test=taskds[runmask2] #classifier testing dataset

        clf.train(taskds_train)
        predictions=clf.predict(taskds_test.samples)
        clf.untrain()
        storer['sub%i' %(sub-1)].append(np.mean(predictions==taskds_test.sa.targets)) #normal withinsub classification accuracy

        wholeallsub.append(wholedataset)
        allsub.append(taskds_train)
        allsubtest.append(taskds_test)

#helped by hyper
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
        #print("doing %i" %me)
        mask=np.isin(my_score,np.sort(my_score)[-60:]) #use 60 voxels
        allsub[me]=allsub[me][:,mask]
        allsubtest[me]=allsubtest[me][:,mask]
        wholeallsub[me]=wholeallsub[me][:,mask]
#end of correlation FeatureSelection,ref:haxby,2011-----------------------------------------------------

    hyper = Hyperalignment()
    hypmaps = hyper(wholeallsub)
    test_hyper = [h.forward(sd) for h, sd in zip(hypmaps, allsubtest)] #test on one subkect's data
    train_hyper= [h.forward(sd) for h, sd in zip(hypmaps,allsub)] #use all subjects' hyperaligned training dataset to train classifier
    train_hyper = vstack(train_hyper)
    test_hyper = vstack(test_hyper)

    clf = LinearCSVMC()
    for sub in range(1,sub_number+1):
        new_test=test_hyper[test_hyper.sa.subject==sub]
        new_train=train_hyper

        clf.train(new_train)
        predictions=clf.predict(new_test.samples)
        clf.untrain()
        hyperstorer['sub%i' %(sub-1)].append(np.mean(predictions==new_test.sa.targets)) #withinsub classification accuracy, using all subjects' data to train

    print("finished loop %i" %hugeiteration)

#accuracy saving
print('within score:')
print('sub0: %f' %np.mean(storer['sub0']))
print('sub1: %f' %np.mean(storer['sub1']))
print('sub2: %f' %np.mean(storer['sub2']))
print('sub3: %f' %np.mean(storer['sub3']))
print('sub4: %f' %np.mean(storer['sub4']))
print('sub5: %f' %np.mean(storer['sub5']))
df0=pd.DataFrame(data={'score':storer['sub0'],'subject':['sub0']*8,'helped by hyperalignment':['No']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
df1=pd.DataFrame(data={'score':storer['sub1'],'subject':['sub1']*8,'helped by hyperalignment':['No']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
df2=pd.DataFrame(data={'score':storer['sub2'],'subject':['sub2']*8,'helped by hyperalignment':['No']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
df3=pd.DataFrame(data={'score':storer['sub3'],'subject':['sub3']*8,'helped by hyperalignment':['No']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
df4=pd.DataFrame(data={'score':storer['sub4'],'subject':['sub4']*8,'helped by hyperalignment':['No']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
df5=pd.DataFrame(data={'score':storer['sub5'],'subject':['sub5']*8,'helped by hyperalignment':['No']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
df=pd.concat([df0,df1,df2,df3,df4,df5],axis=0,ignore_index=True)
df['special']='all 2 sesion'
df.to_csv('/Users/js94538/motoranalysis/output/less_localizer_within_2sess_hyper_%i_run_train_%i_run.txt' %(indihyper,trainnumber))

print('helped by hyper score:')
print('sub0: %f' %np.mean(hyperstorer['sub0']))
print('sub1: %f' %np.mean(hyperstorer['sub1']))
print('sub2: %f' %np.mean(hyperstorer['sub2']))
print('sub3: %f' %np.mean(hyperstorer['sub3']))
print('sub4: %f' %np.mean(hyperstorer['sub4']))
print('sub5: %f' %np.mean(hyperstorer['sub5']))

hdf0=pd.DataFrame(data={'score':hyperstorer['sub0'],'subject':['sub0']*8,'helped by hyperalignment':['Yes']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
hdf1=pd.DataFrame(data={'score':hyperstorer['sub1'],'subject':['sub1']*8,'helped by hyperalignment':['Yes']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
hdf2=pd.DataFrame(data={'score':hyperstorer['sub2'],'subject':['sub2']*8,'helped by hyperalignment':['Yes']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
hdf3=pd.DataFrame(data={'score':hyperstorer['sub3'],'subject':['sub3']*8,'helped by hyperalignment':['Yes']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
hdf4=pd.DataFrame(data={'score':hyperstorer['sub4'],'subject':['sub4']*8,'helped by hyperalignment':['Yes']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})
hdf5=pd.DataFrame(data={'score':hyperstorer['sub5'],'subject':['sub5']*8,'helped by hyperalignment':['Yes']*8,'hyper run number':[indihyper]*8,'train run number':trainnumber})

hdf=pd.concat([hdf0,hdf1,hdf2,hdf3,hdf4,hdf5],axis=0,ignore_index=True)
hdf['special']='all 2 session'
hdf.to_csv('/Users/js94538/motoranalysis/output/less_localizer_allsub_2sess_hyper_%i_run_train_%i_run.txt' %(indihyper,trainnumber))
