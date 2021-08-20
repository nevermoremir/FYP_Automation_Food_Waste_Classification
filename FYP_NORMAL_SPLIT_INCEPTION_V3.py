#This code will run and collect all the information of model performance and save the figure output plot as well as confusion matrix. 
#Please read throughout the code and change the directory based on your computer location save. 
#The datasets must be downlaod from google drive =  https://drive.google.com/drive/folders/1aenzdTMwEeNogVEMEayoKXl7L5eTKM-9?usp=sharing


#library need to be installed
from __future__ import print_function
from __future__ import division
from multiprocessing import Process, freeze_support
from matplotlib import figure
from sklearn import metrics
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import seaborn as sns



if __name__ == '__main__':
    first = time.time()  
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    data_dir = "./normal_split_datasets" # Data directory

    # Models 
    model_name = "inception"

    # Number of classes in the dataset , in this case are landfill and compost
    num_classes = 2

    # Batch size for training (change depending on how much memory)
    batch_size = 25 #comment this if the test_epoch is false
    #batch_size = [10,25] #Uncomment this if the test_epoch is false

    # Number of epochs to train for
    num_epochs = [100]#comment this if the test_epoch is false
    #num_epochs = 30 #Uncomment this if the test_epoch is false

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    #Flag for validationt testing, When False, we test on variation of batchsize,
    #   when True we test on variation of epochs
    test_epoch= True

    #hold data
    #train variable
    b_acc=[]
    b_f1_c=[]
    b_f1_l=[]
    b_precision_c=[]
    b_precision_l=[]
    b_recall_c=[]
    b_recall_l=[]
    b_y_pred=[]
    b_y_actual=[]
    time_take=[]
    #test variable
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
    n_batch=[]
    n_epoch=[]
    finetuning=[]
    


    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time() #start time

        #Variable for hold data collection
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

        best_model_wts = copy.deepcopy(model.state_dict())#Copy best model 
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                #Variable for hold data collection
                running_loss = 0.0
                running_corrects = 0
                y_pred=[]
                y_actual =[]

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
            
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
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
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('Y_Actual : {}'.format(y_actual))
                print('Y_Predict : {}'.format(y_pred))
                target_names =['Compost','Landfill']
                report = classification_report(y_actual,y_pred,target_names=target_names,output_dict=True)
                report_gui = classification_report(y_actual,y_pred,target_names=target_names,output_dict=False)
                print(report_gui)
                print('F1 score compost : {}'.format(round(report['Compost']['f1-score'],2)))
                print('F1 score landfill : {}'.format(round(report['Landfill']['f1-score'],2)))
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print ('{} dataloaders :{}'.format(phase,len(dataloaders[phase].dataset)))
                print('{} running_correct :{}'.format(phase,running_corrects))

                # deep copy the model and data collection
                if phase == 'val' and epoch_acc > best_acc:
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
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        b_y_actual.append(best_y_actual)
        b_y_pred.append(best_y_pred)
        b_acc.append(round(best_acc.item(),6))
        b_f1_c.append(round(best_f1_s_c,6))
        b_f1_l.append(round(best_f1_s_l,6))
        b_precision_c.append(round(best_precision_c,6))
        b_precision_l.append(round(best_precision_l,6))
        b_recall_c.append(round(best_recall_c,6))
        b_recall_l.append(round(best_recall_l,6))
        print("best_y_actual : ",b_y_actual)
        print("best_y_pred : ",b_y_pred)
        time_take.append(str(round(time_elapsed//60)) +'m'+' '+ str(round(time_elapsed%60)) + 's')
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best F1-score-compost: {:4f}'.format(best_f1_s_c))
        print('Best F1-score-landfill: {:4f}'.format(best_f1_s_l))
        print('Best Precision-compost: {:4f}'.format(best_precision_c))
        print('Best Precision-landfill: {:4f}'.format(best_precision_l))
        print('Best Recall-landfill: {:4f}'.format(best_recall_l))
        print('Best Recall-compost: {:4f}'.format(best_recall_c))
      
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def test_model(model,class_name,number_images,dataloaders):
        since =time.time()
        model=model.eval()
        image_handled= 0
        itter=0
        plt.figure()
        test_actual=[]
        test_predict=[]
        test_correct=0.0
        with torch.no_grad():
            for inputs,labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs= model(inputs)
                _, preds = torch.max(outputs,1)
                itter+=1 #itter for save image purpose
                test_correct += torch.sum(preds == labels.data)
                t_to_np_pred= preds.detach().cpu().numpy()
                test_predict =np.append(test_predict,t_to_np_pred,axis=None)
                t_to_np_actual= labels.data.detach().cpu().numpy()
                test_actual = np.append(test_actual,t_to_np_actual,axis=None)
                
                #Save images output
                for j in range(inputs.shape[0]):
                    image_handled+= 1
                    fig, ax =plt.subplots()
                    ax.axis("off")
                    ax.set_title(f'actual : {class_name[labels[j]]} , predicted: {class_name[preds[j]]}')
                    data2=inputs.cpu().data[j].permute(1, 2, 0).numpy()
                    plt.imshow(data2)
                    plt.savefig("C:/Users/SM/Desktop/Code FYP/Test_result images/output_"+str(itter)+"_"+str(j))
                    plt.show()
                    plt.close()

        #data collection
        target_names =['Compost','Landfill']
        report = classification_report(test_actual,test_predict,target_names=target_names,output_dict=True)
        report_gui = classification_report(test_actual,test_predict,target_names=target_names,output_dict=False)
        epoch_acc= report['accuracy']
        print('{}   Acc: {:.4f}'.format("Testing", epoch_acc))
        print(report_gui)
        
        print('F1 score compost : {}'.format(round(report['Compost']['f1-score'],2)))
        print('F1 score landfill : {}'.format(round(report['Landfill']['f1-score'],2)))
        print ('{} dataloaders :{}'.format("Training",len(dataloaders['test'].dataset)))
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
                    

    #Set updated parameter for feature extraction 
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
 #test_epoch to decide on collection data bsed on epoch or batch size changing parameter
 
    if test_epoch == True:
        test_param = num_epochs
    if test_epoch == False:
        test_param = batch_size
    
    for x_param in test_param:

        for y in range(1): #run loop data collection purpose
           
            if test_epoch == False:
                n_batch.append(x_param)
                print('Batch Size = {} , Round = {}'.format(x_param,y+1))
                n_epoch.append(num_epochs)
            if test_epoch == True:
                n_batch.append(batch_size)
                print('Batch Size = {} , Round = {}'.format(x_param,y+1))
                n_epoch.append(x_param)

            finetuning.append(not feature_extract)
           
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
                'test': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val','test']}
            # Create training and validation dataloaders
            #print(image_datasets)

            if test_epoch == True:
                dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val' ,'test']}
            if test_epoch == False:
                dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=x_param, shuffle=True, num_workers=4) for x in ['train', 'val','test']}
            
            
            print(dataloaders_dict)
            # Detect if we have a GPU available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cpu")

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

            #Train and validate based on test_epoch parameter
            if test_epoch == False:
                model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs= num_epochs, is_inception=(model_name=="inception"))
                print('Batch Size = {} , Round = {}'.format(x_param,y+1))
            if test_epoch == True:
                model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs= x_param, is_inception=(model_name=="inception"))
                print('Epochs = {} , Round = {}'.format(x_param,y+1))

            #Confusion Metric
            #Please check the directory for saving confusion matrix
            print('B_y_actual',b_y_actual)
            print('B_y_pred',b_y_pred)
            cm = metrics.confusion_matrix(b_y_actual[y], b_y_pred[y])
            plt.figure(figsize=(9,9))
            categories = ['Compost','Landfill']
            sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r' ,xticklabels=categories , yticklabels=categories)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.title("Best confusion matrix , Epochs = {} Round = {} ".format(x_param,y), size = 15)
            plt.savefig("C:/Users/SM/Desktop/Code FYP/Result image/Normalsplit/Confusion Metric_Training_"+"Number_Epoch_"+str(x_param)+"_"+str(y+1)+".png")

            
            
            #Save model training for future use
            #PATH = "./entire_model_Number_Epoch_"+str(x_param)+"_"+str(y+1)+".pt" #Uncomment this for save file training model on that path, please check directory is true
            # Save
            #torch.save(model_ft, PATH) #saving the model 

            # # Load
            #model_ft = torch.load(PATH) #load from the saving model

            #Testing Model
            test_model(model_ft,['Compost','Landfill'],10,dataloaders_dict)

            #Confusion Metric for testing
            #Please check the directory for saving confusion matrix
            cm = metrics.confusion_matrix(test_y_actual[y], test_y_pred[y])
            plt.figure(figsize=(9,9))
            categories = ['Compost','Landfill']
            sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r' ,xticklabels=categories , yticklabels=categories)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.title("Best confusion matrix , Epochs = {} Round = {} ".format(x_param,y), size = 15)
            plt.savefig("C:/Users/SM/Desktop/Code FYP/Result image/Normalsplit/Confusion Metric_Testing_"+"Number_Epoch_"+str(x_param)+"_"+str(y+1)+".png")

    #Data collection summarisation        
    simulation_time =time.time() - first
    print('Simulation complete in {:.0f}m {:.0f}s'.format(simulation_time // 60, simulation_time % 60))
    simulation=  str(round(simulation_time //60)) +'m'+' '+ str(round(simulation_time % 60)) + 's'
    print(n_epoch)
    print(b_acc)
    print(time_take)

    #Save data collection Validation on EXCEL
    result = pd.DataFrame()
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

    result.to_excel('validation_final.xlsx', index=False)
   
   #Save data collection Testing on EXCEL
    test_result =pd.DataFrame()
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

    test_result.to_excel('Testing_final.xlsx', index=False)




