#This code will run and collect all the information of model performance and save the figure output plot as well as confusion matrix. 
#Please read throughout the code and change the directory based on your computer location save. 
#The datasets must be downlaod from google drive =  https://drive.google.com/drive/folders/1aenzdTMwEeNogVEMEayoKXl7L5eTKM-9?usp=sharing
#Creadit to pytorch framework website as guidance for the code sources


#library need to be installed
from __future__ import print_function
from __future__ import division
from multiprocessing import Process, freeze_support
from numpy.lib.function_base import append
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import ConcatDataset
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import seaborn as sns




if __name__ == '__main__':
    first = time.time()  #start time
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)


 

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
 
    data_dir = "./k_fold_datasets" #Data directory 

    # Models 
    model_name = "inception"

    # Number of classes in the dataset , in this case are landfill and compost
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = 25 #comment this if the test_epoch is false
    #batch_size = [10,25] #Uncomment this if the test_epoch is false

    # Number of epochs to train for
    num_epochs = [20,30] #comment this if the test_epoch is false
    #num_epochs = 30 #Uncomment this if the test_epoch is false

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    #Flag for validationt testing, When False, we test on variation of batchsize,
    #   when True we test on variation of epochs
    test_epoch= True

    #kFOLD
    kfolds =8
    kfold =KFold(n_splits=kfolds, shuffle=True)

    #hold data
    #train data
    b_acc=[]
    b_f1_c=[]
    b_f1_l=[]
    b_precision_c=[]
    b_precision_l=[]
    b_recall_c=[]
    b_recall_l=[]
    b_y_pred=[]
    b_y_actual=[]

    #test data
    test_acc=[]
    test_f1_c=[]
    test_f1_l=[]
    test_precision_c=[]
    test_precision_l=[]
    test_recall_c=[]
    test_recall_l=[]
    test_y_pred=[]
    test_y_actual=[]    
    test_time_take=[]
    time_take=[]
    n_batch=[]
    n_epoch=[]
    n_fold=[]
    finetuning=[]
    


    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time() #Start time
        #Data hold for collection
        val_acc_history = []
        best_acc = 0.0
        best_f1_s_c =0.0
        best_f1_s_l=0.0
        best_precision_c=0.0
        best_precision_l=0.0
        best_recall_c=0.0
        best_recall_l=0.0
        best_y_pred =[]
        best_y_actual=[]

        best_model_wts = copy.deepcopy(model.state_dict()) #copy model with best parameter

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has only training phase
            
               
            model.train()  # Set model to training mode
               
            #Data hold for collection
            running_loss = 0.0
            running_corrects = 0
            y_pred=[]
            y_actual =[]

            # Iterate over data.
            for inputs,labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                    if is_inception :
                       
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                   
                    # backward + optimize only if in training phase
                    
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                t_to_np_pred= preds.detach().cpu().numpy()
                y_pred =np.append(y_pred,t_to_np_pred,axis=None)
                t_to_np_actual= labels.data.detach().cpu().numpy()
                y_actual = np.append(y_actual,t_to_np_actual,axis=None)

            #Debug Purpose
            print('Y_Actual : {}'.format(y_actual))
            print('Y_Predict : {}'.format(y_pred))
            target_names =['Compost','Landfill']
            report = classification_report(y_actual,y_pred,target_names=target_names,output_dict=True) # classification report in dict mode
            report_gui = classification_report(y_actual,y_pred,target_names=target_names,output_dict=False) #classification report in gui mode
            print(report_gui)
            print('F1 score compost : {}'.format(round(report['Compost']['f1-score'],2)))
            print('F1 score landfill : {}'.format(round(report['Landfill']['f1-score'],2)))
            epoch_acc =report['accuracy']
            print('{} Acc: {:.4f}'.format("Training", epoch_acc))
            print ('{} dataloaders :{}'.format("Training",report['weighted avg']['support']))
            print('{} running_correct :{}'.format("Training",running_corrects))


            # deep copy the model and data collection
            if  epoch_acc > best_acc:
                best_acc = epoch_acc
                best_f1_s_c = report['Compost']['f1-score']
                best_f1_s_l = report['Landfill']['f1-score']
                best_precision_c= report['Compost']['precision']
                best_precision_l = report['Landfill']['precision']
                best_recall_c = report['Compost']['recall']
                best_recall_l = report['Landfill']['recall']
                best_y_actual = y_actual
                best_y_pred = y_pred
                best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(epoch_acc)

            

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        b_y_actual.append(best_y_actual)
        b_y_pred.append(best_y_pred)
        b_acc.append(round(best_acc,6))
        b_f1_c.append(round(best_f1_s_c,6))
        b_f1_l.append(round(best_f1_s_l,6))
        b_precision_c.append(round(best_precision_c,6))
        b_precision_l.append(round(best_precision_l,6))
        b_recall_c.append(round(best_recall_c,6))
        b_recall_l.append(round(best_recall_l,6))
        print("best_y_actual : ",b_y_actual)
        print("best_y_pred : ",b_y_pred)
        time_take.append(str(round(time_elapsed//60)) +'m'+' '+ str(round(time_elapsed%60)) + 's')
        print('Best training Acc: {:4f}'.format(best_acc))
        print('Best F1-score-compost: {:4f}'.format(best_f1_s_c))
        print('Best F1-score-landfill: {:4f}'.format(best_f1_s_l))
        print('Best Precision-compost: {:4f}'.format(best_precision_c))
        print('Best Precision-landfill: {:4f}'.format(best_precision_l))
        print('Best Recall-landfill: {:4f}'.format(best_recall_l))
        print('Best Recall-compost: {:4f}'.format(best_recall_c))
      
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def test_model(model,class_name,number_epoch,dataloaders,fold):
        since =time.time()
        model=model.eval()
        
        plt.figure()
        test_actual=[]
        test_predict=[]
        test_correct=0.0
        with torch.no_grad():
            for inputs,labels in dataloaders:
                image_handled= 0
                itter=0
                inputs = inputs.to(device)
                labels = labels.to(device)
                #forward inputs
                outputs= model(inputs)
                _, preds = torch.max(outputs,1)

                #Calculate ground truth and prediction
                test_correct += torch.sum(preds == labels.data)
                t_to_np_pred= preds.detach().cpu().numpy()
                test_predict =np.append(test_predict,t_to_np_pred,axis=None)
                t_to_np_actual= labels.data.detach().cpu().numpy()
                test_actual = np.append(test_actual,t_to_np_actual,axis=None)

                #Save images output prediction
                for j in range(inputs.shape[0]):
                    image_handled+= 1
                    fig, ax =plt.subplots()
                    ax.axis("off")
                    ax.set_title(f'actual : {class_name[labels[j]]} , predicted: {class_name[preds[j]]}')
                    plt.imshow(inputs.cpu().data[j].permute(1, 2, 0).numpy())
                    #Please change the directory to your directory for saving outputs file
                    plt.savefig("C:/Users/SM/Desktop/Code FYP/Test_result_kfold_images/output_"+str(itter)+"_"+str(j)+"_Number epoch_"+str(number_epoch)+"_No_Fold_"+str(fold))
                    plt.close()
                    if image_handled == 10: #comment this if want all the output to be saved
                        break

        
        
        target_names =['Compost','Landfill']
        report = classification_report(test_actual,test_predict,target_names=target_names,output_dict=True)
        report_gui = classification_report(test_actual,test_predict,target_names=target_names,output_dict=False)
        epoch_acc= report['accuracy']
        print('{} Fold: {}  Acc: {:.4f}'.format("Testing",fold, epoch_acc))
        print(report_gui)
        
        print('F1 score compost : {}'.format(round(report['Compost']['f1-score'],2)))
        print('F1 score landfill : {}'.format(round(report['Landfill']['f1-score'],2)))
        print ('{} dataloaders :{}'.format("Training",len(dataloaders.dataset)))
        print('{} running_correct :{}'.format("Training",test_correct))
        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        test_time_take.append(str(round(time_elapsed//60)) +'m'+' '+ str(round(time_elapsed%60)) + 's')
        test_y_actual.append(test_actual)
        test_y_pred.append(test_predict)
        test_acc.append(round(epoch_acc,6))
        test_f1_c.append(round(report['Compost']['f1-score'],6))
        test_f1_l.append(round(report['Landfill']['f1-score'],6))
        test_precision_c.append(round(report['Compost']['precision'],6))
        test_precision_l.append(round(report['Landfill']['precision'],6))
        test_recall_c.append(round(report['Compost']['recall'],6))
        test_recall_l.append(round(report['Landfill']['recall'],6))

        return 





    #Set the paramter to be updated based on feature learning or finetuning 
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False







    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    #Start main code
    #test_epoch to decide on collection data based on epoch or batch size changing parameter

    if test_epoch == True:
        test_param = num_epochs
    if test_epoch == False:
        test_param = batch_size
    
    for x_param in test_param:

        for y in range(1): #run loop data collection purpose
           
            
           
            # Initialize the model for this run
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # Print the model we just instantiated
            print(model_ft)

            # Data augmentation and normalization for training
            # Just normalization for validation
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
            image_datasets = ConcatDataset([image_datasets['train'],image_datasets['val']])

            for fold, (train_ids, val_ids) in enumerate(kfold.split(image_datasets)):
                # Print
                print(f'FOLD {fold}')
                print('--------------------------------')
                n_fold.append(fold)
                if test_epoch == False:
                    n_batch.append(x_param)
                    print('Batch Size = {} , Round = {}'.format(x_param,y+1))
                    n_epoch.append(num_epochs)
                if test_epoch == True:
                    n_batch.append(batch_size)
                    print('Epochs = {} , Round = {}'.format(x_param,y+1))
                    n_epoch.append(x_param)

                finetuning.append(not feature_extract)
                
                # Sample elements randomly from a given list of ids(index), no replacement.
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                print(train_subsampler)
                test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
                print(test_subsampler)
                # Define data loaders for training and testing data in this fold
                trainloader = torch.utils.data.DataLoader(
                                image_datasets, 
                                batch_size=batch_size, sampler=train_subsampler,num_workers=4)
                valloader = torch.utils.data.DataLoader(
                                image_datasets,
                                batch_size=batch_size, sampler=test_subsampler,num_workers=4)
            
                
                # Detect if we have a GPU available
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                #device = torch.device("cpu") #Uncomment this if want cpu 

                # Send the model to GPU
                model_ft = model_ft.to(device)

                # Gather the parameters to be optimized/updated in this run. If we are
                #  finetuning we will be updating all parameters. However, if we are
                #  doing feature extract method, we will only update the parameters
                #  that we have just initialized, i.e. the parameters with requires_grad
                #  is True.
                params_to_update = model_ft.parameters()
                print("Params to learn:")
                if feature_extract:
                    params_to_update = []
                    for name,param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)
                            print("\t",name)
                else:
                    for name,param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            print("\t",name)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

                # Setup the loss fxn    
                criterion = nn.CrossEntropyLoss()


                freeze_support()

                # Train and evaluate based on epoch or batch size for parameter
                if test_epoch == False:
                    model_ft, hist = train_model(model_ft, trainloader, criterion, optimizer_ft, num_epochs= num_epochs, is_inception=(model_name=="inception"))
                    print('Batch Size = {} , Round = {}'.format(x_param,y+1))
                if test_epoch == True:
                    model_ft, hist = train_model(model_ft, trainloader, criterion, optimizer_ft, num_epochs= x_param, is_inception=(model_name=="inception"))
                    print('Epochs = {} , Round = {}'.format(x_param,y+1))
                
                #Confusion Metric for training
                #Please check the directory for saving confusion matrix
                cm = metrics.confusion_matrix(b_y_actual[fold], b_y_pred[fold])
                plt.figure(figsize=(9,9))
                categories = ['Compost','Landfill']
                sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r' ,xticklabels=categories , yticklabels=categories)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                plt.title("Best confusion matrix , Epochs = {} K-Fold = {} ".format(x_param,fold), size = 15)
                plt.savefig("C:/Users/SM/Desktop/Code FYP/Result image/kfold/Confusion Metric_Training_"+"FOLD_"+str(fold)+"Number_Epoch_"+str(x_param)+"_"+str(y+1)+".png")
                
                #Testing
                test_model(model_ft,['Compost','Landfill'],x_param,valloader,fold)

                #Confusion Metric for testing
                #Please check the directory for saving confusion matrix
                cm = metrics.confusion_matrix(test_y_actual[fold], test_y_pred[fold])
                plt.figure(figsize=(9,9))
                categories = ['Compost','Landfill']
                sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r' ,xticklabels=categories , yticklabels=categories)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                plt.title("Best confusion matrix , Epochs = {} K-Fold = {} ".format(x_param,fold), size = 15)
                plt.savefig("C:/Users/SM/Desktop/Code FYP/Result image/Kfold/Confusion Metric_Testing"+"FOLD_"+str(fold)+"Number_Epoch_"+str(x_param)+"_"+str(y+1)+".png")
                
                

    #data summarisation at the end of loop       
    simulation_time =time.time() - first
    print('Simulation complete in {:.0f}m {:.0f}s'.format(simulation_time // 60, simulation_time % 60))
    simulation=  str(round(simulation_time //60)) +'m'+' '+ str(round(simulation_time % 60)) + 's'
    print(n_epoch)
    print(b_acc)
    print(time_take)

    #Create dataframe for data collection training
    result = pd.DataFrame()
    result['number_fold'] = n_fold
    result['number_batch']= n_batch
    result['number_epoch']= n_epoch
    result['time_taken'] = time_take
    result['finetuning']= finetuning
    result['best_accuracy']= b_acc
    result['best_precision compost'] =b_precision_c
    result['best_precision landfillt'] =b_precision_l
    result['best_recall compost'] = b_recall_c
    result['best_recall landfill'] = b_recall_l
    result['best_f1-score compost']=b_f1_c
    result['best_f1-score landfill']= b_f1_l
    result['Simulation Time'] = simulation

    result.to_excel('training_k-fold_version_5_'+str(kfolds)+'.xlsx', index=False) #creating excel file, make sure the directory is true

    #Create dataframe for data collection testing
    test_result =pd.DataFrame()
    test_result['number_fold'] =n_fold
    test_result['number_batch'] =n_batch
    test_result['number_epoch']= n_epoch
    test_result['time_taken'] = test_time_take
    test_result['finetuning']= finetuning
    test_result['accuracy']= test_acc
    test_result['precision compost'] =test_precision_c
    test_result['precision landfillt'] =test_precision_l
    test_result['recall compost'] = test_recall_c
    test_result['recall landfill'] = test_recall_l
    test_result['f1-score compost']=test_f1_c
    test_result['f1-score landfill']= test_f1_l
    test_result['Simulation Time'] = simulation

    test_result.to_excel('testing_k-fold_version_5_'+str(kfolds)+'.xlsx', index=False) #creating excel file, make sure the directory is true


   





