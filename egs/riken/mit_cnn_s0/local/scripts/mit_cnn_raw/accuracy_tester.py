'''
Tests the accuracy of a bunch of predictions compared to ground truth labels
The script loops through the prediction files made using each cutoff to print out accuracy metrics
'''
import numpy as np
import sklearn.metrics


for num in [0,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
    prediction_list=['results/20161219_Athos_model_small_{}%.txt'.format(int(100*num)),
                     'results/20161219_Porthos_model_small_{}%.txt'.format(int(100*num))]
    
    correct_list=["Wave_files/20161219_Athos_Porthos/Athos_20161219.txt",
                  "Wave_files/20161219_Athos_Porthos/Porthos_20161219.txt"]
    
    classes=['chi','cha','ek','noise','ot','ph','tr','trph','ts','tw']
    
    '''initiates a lists of several accuracy metrics, adding a 1 for each correct and
    a zero for each error
    
    noise_correct measures accuracy on cases where there is no call according to human labels
    
    signal_correct measures accuracy when there is a call labeled by humans'''
    noise_correct=[]
    signal_correct=[]

    #loops through each of the prediction files individually
    for k in range(len(prediction_list)):
        predictions=open(prediction_list[k],'r')
        correct=open(correct_list[k],'r')

        '''Makes lists of the labels based on the text files since lists are easier to compare
        by discretizing the labels into chunks of 50ms'''
        lines_pred=[]
        '''current keeps track of the start of our window, start time can be calculated as 50ms*current'''
        current=0
        for line in predictions:
            start_t, end_t, typee=line.split('\t')
            '''if the start of the next call is more than 274ms away from the start of our window
            there is no call in the middle 50ms of our 500ms window'''
            while float(start_t)-current*0.05>0.274:
                current+=1
                lines_pred.append('noise')
            '''after a call ihas started, as long as it ends at more than 226ms from the start of our window
            a call can be considered to be at the middle 50ms of our window. After it passes that we loop to next
            call in the prediction file'''
            while float(end_t)-0.05*current>0.226:
                lines_pred.append(typee[:-1])
                current+=1
                
        lines_corr=[]
        current=0
        '''keeps track of where first human label occurred in order to only compare accuracy after that point
        since calls before a certain point were not labeled in the experiment'''
        first=None
        for line in correct:
            start_t, end_t, typee=line.split('\t')
            typee=typee.lower()
            while float(start_t)-current*0.05>0.274:
                current+=1
                lines_corr.append('noise')
            while float(end_t)-0.05*current>0.226:
                if first==None:
                    first=current
                if typee[:-1] not in classes:
                    lines_corr.append('noise')
                else:
                    lines_corr.append(typee[:-1])
                current+=1
                
        last=len(lines_corr)
        lines_list=[lines_pred,lines_corr]
        'In case predictions list ends before correct list ends we pad it with noise until they have the same size'
        if len(lines_pred)<last:
            for i in range(last-len(lines_pred)):
                lines_pred.append('noise')
                    

        for j in range(first,last):
            #No call
            if lines_corr[j]=='noise':
                if lines_pred[j]=='noise':
                    noise_correct.append(1)  
                else:
                    noise_correct.append(0)
            #Call   
            elif lines_corr[j]!='noise':
                if lines_corr[j]==lines_pred[j]:
                    signal_correct.append(1)
                else:
                    signal_correct.append(0)
                

    total=np.concatenate((signal_correct,noise_correct))
    print("Cutoff:{}".format(num))
    #prints out classification accuracy in case of call, no call and total
    to_print="Fraction correctly classified: Noise:{:.4f},Call: {:.4f} , Total:{:.4f}"
    print(to_print.format(np.mean(noise_correct),np.mean(signal_correct),np.mean(total)))
    #Counts false positives by counting the amount of mistakes when the correct label is noise
    false_positives=0
    for i in noise_correct:
        if i==0:
            false_positives+=1
    #Sum of signal_correct is the amount of true positives
    precision=np.sum(signal_correct)/(np.sum(signal_correct)+false_positives)
    #recall is just TP/(TP+FN) which is the same as mean of signal_correct
    recall=np.mean(signal_correct)
    f1=2*recall*precision/(recall+precision)
    
    print("Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}".format(recall,precision,f1))
    
