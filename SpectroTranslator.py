###################################################
### MAIN CODE OF THE SPECTROTRANSLATOR SOFTWARE ###
###                                             ###
### By G.F. THOMAS Nov.2023                     ###
###################################################
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from astropy.modeling import models, fitting
import math
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from os import path
import sys
import pandas as pd

# For sklearn
from sklearn.preprocessing import StandardScaler


## Others
import shap
import pickle



# For Keras
import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from keras.models import Model
from keras.layers import Input,Layer, Dense, BatchNormalization, Dropout, Activation, multiply
from keras import backend as K
from keras.callbacks import EarlyStopping

    

# My Own supercool colorbar
import matplotlib.colors as mcolors
my_map = ['#6F4C9B','#6059A9','#5568B8','#4E79C5','#4D8AC6','#4E96BC','#549EB3','#59A5A9','#60AB9E','#69B190','#77B77D','#8CBC68','#A6BE54','#BEBC48','#D1B541','#DDAA3C','#E49C39','#E78C35','#E67932','#E4632D','#DF4828','#DA2222','#B8221E']
SRON= mcolors.LinearSegmentedColormap.from_list('SRON',my_map)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"]=12
    
    
    
## To set the seed
def set_randomseed(seed):
	np.random.seed(seed)
	tf.random.set_seed(seed)
    
# Class to store the normalization parameters use to normalise the input 
def in_poly(x,y,xp,yp):

    tiny=1.e-5;
    simag=np.zeros(len(x))

    for j in range(0,len(xp)):
        if (j<(len(xp)-1)):
            xe=xp[j+1]
            xs=xp[j]
            ye=yp[j+1]
            ys=yp[j]
        else:
            xe=xp[0]
            xs=xp[j]
            ye=yp[0]
            ys=yp[j]
        simag+=fimag(x,xs,xe,y,ys,ye)


    simag[(np.abs(simag)>tiny)]=1.0
    simag[(np.abs(simag)<=tiny)]=0.0
    return simag



def fimag(x0, xs,xe, y0,  ys,  ye):

    top= -(xe-x0) * (ys-y0) + (ye-y0) * (xs-x0)
    bot=  (xe-x0) * (xs-x0) + (ye-y0) * (ys-y0)
    return np.arctan2(top,bot)

### Gaussian function
def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5* ((x-mu)/sigma)**2.0 )









################################
### Machine Learning section ###
################################

## Define the ResNet block
#-----------------------------
def ResNet_block(ins,nf=64): 
    ## See ActionFinder. https://arxiv.org/pdf/2012.05250.pdf

    x = Activation('relu')(ins)
    if(nf>=256):
        x=Dropout(0.5)(x)
    x = WeightNormalization(Dense(nf))(x)

    x = Activation('relu')(x)
    if(nf>=256):
        x=Dropout(0.5)(x)
    x = WeightNormalization(Dense(nf))(x)

    return x+ins


## Make the CapNet structure
#-----------------------------
def CapNet(inputs , out_shape=1 ,depth=3,st_features=64,verbose=True,extrinsic=False):

    # Check that the ML is big enough for traning
    assert st_features >= out_shape, "st_features too small"

    ## Define the ANN 
    # Do first linear layer to go from input shape to st_features
    x = Dense(st_features)(inputs)

    # Do a succession of layer a factor 2 larger than previous one
    for i in range (depth):
        nf=st_features*2**i
        nf_nextlayer=st_features*2**(i+1)
        if (verbose):
            print ("depth:= {0}, width: {1}, target: {2}".format(i+1,nf,nf_nextlayer))
        x=ResNet_block(x,nf)
        x = Dense(nf_nextlayer)(x) # Linear layer to prepare the deeper layer

    # Wider layer of the ANN
    if (verbose):
        print ("Central layer, width: {1}, target: {2}".format(i+1,nf_nextlayer,nf_nextlayer))
    x=ResNet_block(x,nf_nextlayer)

    # Do a symetric decreasing in the number of features
    for i in range (depth,0,-1):
        nf=st_features*2**i
        nf_nextlayer=st_features*2**(i-1)
        if (verbose):
            print ("depth:= {0}p, width: {1}, target: {2}".format(i,nf,nf_nextlayer))
        x = Dense(nf_nextlayer)(x) # Linear layer to prepare the current layer
        x=ResNet_block(x,nf_nextlayer)
       

    # Final connected layer to give the shape of the final value
    pred = Dense(out_shape)(x)
    
    # Add the result of the Capnet to the input parameters
    if(extrinsic==False):
        pred=tf.math.add(inputs[:,:out_shape], pred, name=None)
    else: # If parameter is extrinsic
        pred=tf.math.add(tf.reshape(inputs[:,-1],shape=(-1,1)), pred, name=None)

    return pred



## Define the network used by the SpectroTranslator
#---------------------------------------------
def network(data_in,data_out,features_to_learn,depth=3,st_features=64,extrinsic=False,verbose=True):
    """Translate Spectroscopic parameters from one dataset to another
    data_in: inputs of the ANN
    data_out: value of that the ANN should recover 
    features_to_learn: name of the features to be learned
    depth: number of layer of the ResNet
    st_features: number of feature in the based ResNet
    extrinsic: Boolean to know if the parameter to train is the extrincsic network
    verbose: Print all information
    """

    # Define the shape of the input and output layers
    in_shape = data_in.shape[1]
    out_shape = data_out.shape[1]

    # Check that the ML is big enough for traning
    assert st_features >= in_shape-1, "st_features too small"
    assert st_features >= out_shape, "st_features too small"

    # Define the ANN
    inputs = Input(shape=(in_shape,)) # Input layer; Kept as it until the end of the function
    updated_inputs= tf.identity(inputs) # Copy the inputs layer. it then will be update for each featue learned

    pred_final=CapNet(updated_inputs,out_shape=out_shape ,depth=depth,st_features=st_features,extrinsic=extrinsic) 

    print(" ")
    return Model(inputs=inputs, outputs=pred_final,name="SpectroTranslator")
 


## Function for Keras and for the Loss function
def set_lr(model,learning_rate):
    K.set_value(model.optimizer.learning_rate,learning_rate)
    return model


#############################################
## Function to train the SpectroTranslator ##
#############################################
def train_SpectroTranslator(inputs_data_train,outputs_data_train,inputs_data_test,outputs_data_test,features_to_learn,learning_rate,es,optimizer,epochs,batch_size,loss_func="MAE",depth=3,st_features=64,extrinsic=False,name_lossfile="loss.asc",name_plotloss=None):
    """
    Train the SpectroTranslator.
    
    Parameters
    ----------
    inputs_data_train: [pandas DataFrame] Contain the inputs of the training sample
    outputs_data_train: [pandas DataFrame] Contain the outputs of the training sample
    inputs_data_test: [pandas DataFrame] Contain the inputs of the testing(validation) sample
    outputs_data_test: [pandas DataFrame] Contain the outputs of the testing(validation) sample
    features_to_learn: [list of string] Contain the name of the column that contain the parameter to be learned
    learning_rate: [list of float] Containing the different learning rate to train the network
    es: [keras.callbacks.EarlyStopping]: Early stopping criteria
    optimizer: [keras.optimizer] Optimizer used
    epochs: [int] Number of epoch to train the SpectroTranslator
    batch_size: [int] Size of each batches
    loss_func: [str] Name of the Keras loss function to use. Default="MAE"
    depth: [int] Depth of the spectroTranslator. Default=3
    st_features: [int] Number of feature in the 1 depth layer. Default=64
    extrinsic: [bool] If true, it train the extrinsic network, else the intrinsic one. Default=False
    name_lossfile: [str] Name of the file where to store the loss function. Default="loss.asc"
    name_plotloss: [str] Name of the file where to store the plot of the loss function. Default=None

    Return the train network [keras.model] (ANN)
    """
    
    epoch_change_lr=np.zeros(len(learning_rate)-1) # To store the epoch when we change the LR
    # Define the ANN and compile it
    ANN= network(inputs_data_train,outputs_data_train,features_to_learn=features_to_learn,depth=depth,st_features=st_features,extrinsic=extrinsic,verbose=True)
    ANN.compile(optimizer=optimizer, loss=loss_func)

    #keras.utils.plot_model(ANN, "my_first_model.png")
    for idx,lr in enumerate(learning_rate):
        print("Learning Rate: %.g" %lr)
        ANN = set_lr(ANN,lr) # Set the learning rate 
        if (idx>0):
            history = ANN.fit(inputs_data_train,outputs_data_train,
                             epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es],
                             validation_data=(inputs_data_test,outputs_data_test),initial_epoch=int(epoch_change_lr[idx-1]),use_multiprocessing=True)
        else:
            history = ANN.fit(inputs_data_train,outputs_data_train,
                             epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es],
                             validation_data=(inputs_data_test,outputs_data_test),use_multiprocessing=True)


        # If not last lr tested remove epochs after early stopping from the history
        if idx<(len(learning_rate)-1):
            hist_loss=history.history['loss'][:-es.patience-1]
            hist_valloss=history.history['val_loss'][:-es.patience-1]
            epoch_change_lr[idx]=history.epoch[-es.patience]

        else:
            hist_loss=history.history['loss']
            hist_valloss=history.history['val_loss']

        # Save the loss and valloss function
        if(idx>0):
            loss_ANN=np.concatenate([loss_ANN,hist_loss])
            valloss_ANN=np.concatenate([valloss_ANN,hist_valloss])
        else:
            loss_ANN=hist_loss
            valloss_ANN=hist_valloss

    # Save it in a file
    out=np.zeros((len(loss_ANN),3))
    out[:,0]=np.linspace(1,len(loss_ANN),len(loss_ANN))
    out[:,1]=loss_ANN
    out[:,2]=valloss_ANN
    np.savetxt(name_lossfile,out,header="epoch train val",delimiter=" ")
    del(out)

    ## Plot loss function
    plt.plot(loss_ANN,"-",label="Train")
    plt.plot(valloss_ANN,"-",label="Validation")
    for i in range(len(epoch_change_lr)):
        plt.axvline(x=epoch_change_lr[i],ls="--",color="k")

    legend=plt.legend(scatterpoints=1,loc='upper right',fontsize=12)
    legend.get_frame().set_edgecolor('none')
    plt.xlabel(r'Epochs',fontsize=12)
    plt.ylabel(r'Loss',fontsize=12)
    #plt.xscale('log')
    plt.yscale('log')
    if name_plotloss is not None:
        plt.savefig(name_plotloss,format="pdf")
    plt.show()
    plt.close()

    return ANN


##################################################################
## Get the transform values from the trainned SpectroTranslator ##
##################################################################
def get_TransformedValues(data,ANN,inputs,features_to_learn,scaler_out):
    """
    Obtain the Transformed values and store them in the data Frame as featured_pred

    Parameters:
    -----------------------
    data: [pandas.dataFrame] Input dataframe
    ANN: [keras.model] Trainned spectroTranslator model
    inputs: [pandas.dataFrame] Contain the normalized input values
    features_to_learn: [list of str] Contain the name of the learned parameter 
    scaler_out: [sklearn scaler] Trained output scaler 

    Return the input dataframe with predicted values
    """

    ## Get the predicted values and store them in the data dataFrame
    tmp=scaler_out.inverse_transform(ANN.predict(inputs))
    for i,feature in enumerate(features_to_learn):
        data[feature+"_pred"]=tmp[:,i]
    del(tmp)


########################################################
## Preparing training/testing sets and normalize them ##
########################################################
def prepare_dataset_training(train,test,inputs_features,outputs_features):
    # create scaler
    scaler_in = StandardScaler()
    # apply transform
    inputs_data_train = scaler_in.fit_transform(train[inputs_features])
    inputs_data_test = scaler_in.transform(test[inputs_features])


    scaler_out = StandardScaler()
    outputs_data_train=scaler_out.fit_transform(train[outputs_features])
    outputs_data_test=scaler_out.transform(test[outputs_features])
    return inputs_data_train,outputs_data_train,inputs_data_test,outputs_data_test,scaler_in,scaler_out


def prepare_dataset_applying(train,test,inputs_features,outputs_features,scaler_in_file,scaler_out_file):
    ## Check that the scaler files exist
    if(path.exists(scaler_in_file) and path.exists(scaler_out_file)):
       scaler_in = pickle.load(open(scaler_in_file,'rb'))
       scaler_out = pickle.load(open(scaler_out_file,'rb'))
    else:
       sys.exit("ERROR:: "+scaler_in_file+" and/or "+scaler_out_file+" does not exist")
    
    # apply transform
    inputs_data_train = scaler_in.transform(train[inputs_features])
    inputs_data_test = scaler_in.transform(test[inputs_features])
    outputs_data_train=scaler_out.transform(train[outputs_features])
    outputs_data_test=scaler_out.transform(test[outputs_features])
    return inputs_data_train,outputs_data_train,inputs_data_test,outputs_data_test,scaler_in,scaler_out




################################################
## MONTE-CARLO sampling of the inputs/outputs ##
################################################
def montecarlo(train,test,inputs_features,outputs_features,error_inputs,error_outputs,Nreal=10):
    # For the training set
    for real in range(0,Nreal):
        tmp1=train.copy()
        tmp2=test.copy()
        tmp1['real']=real*np.ones(len(train))
        tmp2['real']=real*np.ones(len(test))

        if(real>0):
            for count,feature in enumerate(inputs_features): # inputs features
                if(error_inputs[count]!= ""):
                    tmp1[feature]=train[feature]+train[error_inputs[count]]*np.random.normal(0,1,len(train))
                    tmp2[feature]=test[feature]+test[error_inputs[count]]*np.random.normal(0,1,len(test))
                else:
                    tmp1[feature]=train[feature]
                    tmp2[feature]=test[feature]

            for count,feature in enumerate(outputs_features): # outputs features
                if(error_outputs[count]!= ""):
                    tmp1[feature]=train[feature]+train[error_outputs[count]]*np.random.normal(0,1,len(train))
                    tmp2[feature]=test[feature]+test[error_outputs[count]]*np.random.normal(0,1,len(test))
                else:
                    tmp1[feature]=train[feature]
                    tmp2[feature]=test[feature]
        if(real==0): # Concat the different realizations
            tmp_train=tmp1
            tmp_test=tmp2 
        else:
            tmp_train=pd.concat([tmp_train,tmp1])
            tmp_test=pd.concat([tmp_test,tmp2])
        
    return tmp_train,tmp_test



########################################
## MONTE-CARLO of the applied dataset ##
########################################
def montecarlo_apply(stars,inputs_features,error_inputs,Nreal=10):

    for real in range(0,Nreal):
        tmp1=stars.copy()
        tmp1['real']=real*np.ones(len(stars))

        if(real>0):
            for count,feature in enumerate(inputs_features): # inputs features
                if(error_inputs[count]!= ""):
                    tmp1[feature]=stars[feature]+stars[error_inputs[count]]*np.random.normal(0,1,len(stars))
                else:
                    tmp1[feature]=stars[feature]

        if(real==0): # Concat the different realizations
            tmp=tmp1
        else:
            tmp=pd.concat([tmp,tmp1])
        
    return tmp



##################################
## Apply SpectroTranslator ANNs ##
##################################
def apply_ANNs(data,ANN,inputs_features,outputs_features,scaler_in,scaler_out,Nb_ANN):
    """
    data: Pandas DataFrame containing the full dataset
    ANN: dictionary containing the different ANNs learned
    inputs_features: array containing the name of the inputed features
    outputs_features: array containing the name of the output features
    scaler_in: Scaler function on the input values
    scaler_out: Scaler function on the output values
    Nb_ANN: Number of ANNs trained to be used to estimate the parameters on the uncertaitnties caused by the relations
    
    return data
    """
    
    
    # Rescale the data input
    inputs=scaler_in.transform(data[inputs_features])


    # Put a zero to input values with nan and add a flag
    data['Flag_missing_inputs']=False
    for count,feature in enumerate(inputs_features):
        data['Flag_missing_inputs']=np.where(np.isfinite(inputs[:,count]),data['Flag_missing_inputs'],True)
        inputs[:,count]=np.where(np.isfinite(inputs[:,count]),inputs[:,count],0.0)
        
    
    
    # Create a tables to store the predicted tables
    predicted_values={}
    for feature in outputs_features:
        predicted_values[feature]=np.zeros((len(data),Nb_ANN))

    # Applied the different ANNs
    for i in range(1,Nb_ANN+1):
        tmp_pred=scaler_out.inverse_transform(ANN[str(i)].predict(inputs))
        # store the values in the pred tables
        for j,feature in enumerate(outputs_features):
            predicted_values[feature][:,i-1]=tmp_pred[:,j]
        del tmp_pred
    del(inputs) 


    # Remove the two extremum values to estimate the uncertainties caused by the fitted relation
    for feature in outputs_features:
        predicted_values[feature]=np.sort(predicted_values[feature],axis=1)[:,1:-1] 

        data[feature]=np.mean(predicted_values[feature],axis=1)
        data[feature+"_ERR"]=np.std(predicted_values[feature],axis=1)
        del(predicted_values[feature])
    del(predicted_values)
    
    return data


#################
## SHAP Values ##
#################
def compute_SHAP(ANN,inputs_data_train,inputs_data_test,Nbackground=50,Ntest=100,nsamples=100):
    """
    The SHAP values are used to compute the relative importance of each inputs features
    
    Parameters
    ----------
    ANN: trained Keras ANN model
    input_data_train: pandas DataFrame, contain the normalized input paramaters fo the training set
    input_data_test: pandas DataFrame, contain the normalized input paramaters fo the validation set
    Nbackground: int, number of stars to be use to define the SHAP Kernel
    Ntest: int, Number of stars to select fro mthe validation sample to compute the mean absolute shap values
    nsamples: int, Number of permutation used to compute the individual SHAP values
    """
    background_data=inputs_data_train[np.random.choice(inputs_data_train.shape[0], Nbackground, replace=False), :]
    test_data=inputs_data_test[np.random.choice(inputs_data_test.shape[0], Ntest, replace=False), :]

    explainer = shap.KernelExplainer(model = ANN.predict, data = background_data, link = "identity")
    shap_values=explainer.shap_values(X= test_data, nsamples=nsamples)
    return np.mean(np.abs(shap_values),axis=1) # return the mean of the abs(shap_values)



####################
## Analysis plots ##
####################
# Plot feature importances
def plot_features_importances(results, category_names,namefile_plot=""):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    namefile_plot : str of the namefile to savethe figure. If empty (""), the figure is not saved
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('viridis')(
        np.linspace(0.15, 0.95, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12.2, 6))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    max_data=np.sum(data, axis=1)
    shift=max_data*0.001
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = (data[:, i])
        starts = data_cum[:, i] - widths+shift
        ax.barh(labels, widths-2*shift, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + (widths-4*shift) / 2

        r, g, b, _ = color
        text_color = 'white' if g < 0.8 else 'darkslategrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if(int(c)>0):
                ax.text(x, y, str(round(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    fig.tight_layout()
    if(namefile_plot!=""):
        plt.savefig(namefile_plot, format='pdf')
    #plt.show()
    plt.close()




## Plot the residual for each parameters
def analysis_plot(train,test,inputs_features,outputs_features,error_outputs_features,namefile_plot=""):
    """
    Parameters
    ----------
    train : Pandas DataFrame 
        Contain the training sample
    test : Pandas DataFrame 
        Contain the test(validation) sample
    inputs_features : list of str
        Contain the column name of the input features
    outputs_features : list of str
        Contain the column name of the features to be learned
    error_outputs_features: list of str
        Contain the column name of the error on the features to be learned
    namefile_plot : str of the namefile to savethe figure. If empty (""), the figure is not saved
    """
    fig = plt.figure(figsize=(10.0, 3*len(outputs_features)))
    gs = GridSpec(len(outputs_features),5)

    # To get different color
    color=["rebeccapurple","red","royalblue","darkgoldenrod","green",'navy']
    for count,feature_to_learn in enumerate(outputs_features):

        error_out_feat2learn=error_outputs_features[count]
        
        # Make the second pannel with histogram and a gaussian fit
        ax2 = fig.add_subplot(gs[count,4])
        Nsigma_plot=10 # Number of stddev used to compute the precision of the ML
        stddev=np.std(test[feature_to_learn+'_pred']-test[feature_to_learn])
        Nbins=100
        Nhisto_input, _=np.histogram(test[inputs_features[count]]-test[feature_to_learn],bins=np.linspace(-Nsigma_plot*stddev,Nsigma_plot*stddev,num=Nbins+1),density=False)
        Nhisto, bins=np.histogram(test[feature_to_learn+'_pred']-test[feature_to_learn],bins=np.linspace(-Nsigma_plot*stddev,Nsigma_plot*stddev,num=Nbins+1),density=False)
        Nhisto_input=Nhisto_input/Nhisto.max()       
        Nhisto=Nhisto/Nhisto.max() # Normalize to have the max at 1

        bins_c=bins[:-1]+0.5*(bins[1]-bins[0]) # Get the center of the bins

        
        # Plot the histogram
        ax2.barh(bins_c,Nhisto,height=bins[1]-bins[0],color=color[count],alpha=0.6)

        ## Do the gaussian fitting
        gauss_model = models.Gaussian1D(amplitude=1.0, mean=0, stddev=stddev)
        fit_g = fitting.LevMarLSQFitter(); gauss = fit_g(gauss_model, bins_c, Nhisto)
        ampl=np.max(Nhisto)#gauss.amplitude.value
        mu=np.mean(test[feature_to_learn+'_pred']-test[feature_to_learn])#gauss.mean.value
        sigma=np.std(test[feature_to_learn+'_pred']-test[feature_to_learn])#gauss.stddev.value
        xplot=np.linspace(-Nsigma_plot*sigma,Nsigma_plot*sigma,1000)
        ax2.plot(gauss(xplot),xplot,'-',lw=2,color="k")
        ax2.axhline(y=0,color='k',lw=0.75,alpha=0.3)
        ax2.set_xlim(0,1.2)
        
        #ax2.step(Nhisto_input,bins_c,color="grey",alpha=1.0,lw=1)

        
        
        ## Get the limit for the yplots assuming a homemade receipts
        if(sigma>100):
            val_border_plot_y=math.ceil((5*sigma)/100)*100
            yticks=500
            val_ticks_y=math.ceil(val_border_plot_y/500.0)*500
        elif(sigma>10):
            val_border_plot_y=math.ceil((5*sigma)/100)*100
            yticks=250
            val_ticks_y=val_border_plot_y
        elif(sigma<0.55):
            val_border_plot_y=math.ceil((5*sigma)/0.5)*0.5
            yticks=0.5
            val_ticks_y=val_border_plot_y

        else:
            val_border_plot_y=math.ceil((5*sigma)/0.5)*0.5
            yticks=1.0
            val_ticks_y=val_border_plot_y

                    
        ## Same for the x value
        if((train[feature_to_learn].max()-train[feature_to_learn].min())//100>1):
            x_min_plot=math.floor(train[feature_to_learn].min()/500)*500
            x_max_plot=math.ceil(train[feature_to_learn].max()/500)*500
            xticks=250
        elif((train[feature_to_learn].max()-train[feature_to_learn].min())//2>1):
            x_min_plot=math.floor(train[feature_to_learn].min()/0.2)*0.2
            x_max_plot=math.ceil(train[feature_to_learn].max()/0.2)*0.2
            xticks=0.2
        else:
            x_min_plot=math.floor(train[feature_to_learn].min()/0.1)*0.1
            x_max_plot=math.ceil(train[feature_to_learn].max()/0.1)*0.1
            xticks=0.1
        if feature_to_learn=="MG_FE_APOGEE":
            x_min_plot=math.floor(train[feature_to_learn].min()/0.1)*0.1
            x_max_plot=math.ceil(train[feature_to_learn].max()/0.1)*0.1
            xticks=0.05
            
        ######################################################
        # Make first panel to see the residuals
        ax1 = fig.add_subplot(gs[count,0:4])
        
        # Get the uncertainties in different bins of the feature to learn
        features_means,bins_edges,_=stats.binned_statistic(train[feature_to_learn], train[feature_to_learn+'_pred']-train[feature_to_learn], statistic='mean',
                       bins=np.arange(x_min_plot-xticks/2,x_max_plot+xticks/2*1.001,xticks/2))
        features_stddev,_,_=stats.binned_statistic(train[feature_to_learn], train[feature_to_learn+'_pred']-train[feature_to_learn], statistic='std',
                       bins=bins_edges)
        features_count,_,_=stats.binned_statistic(train[feature_to_learn], train[feature_to_learn+'_pred']-train[feature_to_learn], statistic='count',
                       bins=bins_edges)
        
        # Put to nan bins with not enough data
        Ncut=5
        features_stddev=np.where(features_count>=Ncut,features_stddev,np.nan)
        features_means=np.where(features_count>=Ncut,features_means,np.nan)            
        bins_c=bins_edges#[:-1]+0.5*(bins_edges[1]-bins_edges[0])
        # Fill the values
        features_stddev=np.append(features_stddev,features_stddev[-1])
        features_means=np.append(features_means,features_means[-1])  
        features_count=np.append(features_count,features_count[-1])  

        #for i in range(0,len(bins_c)): # to Fill the values in the upper bin limit 
        #    if(features_count[i]==0 and features_count[i-1]>0):
        #        features_means[i]=features_means[i-1]
        #        features_stddev[i]=features_stddev[i-1]
        for i in range(0,len(bins_edges)-1):
            bins_c=np.array([bins_edges[i],bins_edges[i+1]])
            features_means_c=np.array([features_means[i],features_means[i]])
            features_stddev_c=np.array([features_stddev[i],features_stddev[i]])

            ax1.fill_between(bins_c,features_means_c-features_stddev_c,features_means_c+features_stddev_c,color=color[count],alpha=0.3)
            ax1.fill_between(bins_c,features_means_c-2*features_stddev_c,features_means_c+2*features_stddev_c,color=color[count],alpha=0.2)
            ax1.fill_between(bins_c,features_means_c-3*features_stddev_c,features_means_c+3*features_stddev_c,color=color[count],alpha=0.1)
        ax1.scatter(test[feature_to_learn],test[feature_to_learn+'_pred']-test[feature_to_learn],s=0.5,alpha=1.0,c="k",rasterized=True)        
        ax1.axhline(y=0,color='k',lw=0.75)
        ax1.axhline(y=-3*sigma,color='k',ls="--",lw=0.75)
        ax1.axhline(y=3*sigma,color='k',ls="--",lw=0.75)
        #ax1.axvline(x=train[feature_to_learn].min(),color='k',ls="--",lw=0.75)
        #ax1.axvline(x=train[feature_to_learn].max(),color='k',ls="--",lw=0.75)

        
        ax1.set_xlabel(feature_to_learn,fontsize=12)
        ax1.set_ylabel(r"$\Delta $ "+feature_to_learn,fontsize=12)

        
        ax2.errorbar(0.8,-0.8*val_border_plot_y,yerr=np.mean(train[error_out_feat2learn]),color=color[count],elinewidth=2,capsize=2)

        ##########################################################################
        ## Plot style    
        ax2.set_ylim(-val_border_plot_y-yticks/5,val_border_plot_y+yticks/5)
        ax1.set_ylim(-val_border_plot_y-yticks/5,val_border_plot_y+yticks/5)
        ax1.set_xlim(x_min_plot,x_max_plot)
        #ax1
        minor_locator = AutoMinorLocator(5)
        ax1.xaxis.set_minor_locator(minor_locator)
        ax1.tick_params(which='both', direction='in')
        minor_locatory = AutoMinorLocator(5)
        ax1.yaxis.set_minor_locator(minor_locatory)

        #ax2
        yticks2 = [0,0.5,1]
        tickLabels = map(str, yticks2)
        ax2.set_xticks(yticks2)
        ax2.set_xticklabels(tickLabels)
        ax2.xaxis.set_minor_locator(plt.MaxNLocator(13))
        ax2.yaxis.set_minor_locator(minor_locatory)

        ax2.yaxis.set_ticks_position("right") 
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.tick_params(which='both', direction='in')
        
        ax2.annotate(r"$\mu=$ %.3f"%mu+"\n"+"$\sigma$= %.3f" %(np.abs(sigma)), xy=(0.2,0.8), xycoords="axes fraction", color='black',size=10)



    fig.tight_layout()
    if(namefile_plot!=""):
        plt.savefig(namefile_plot, format='pdf')
    else:
        plt.show()
    plt.close()

    
    
## Plot the residual for each parameters
def analysis_plot_original(train,test,inputs_features,outputs_features,ylabel=None,namefile_plot=""):
    """
    Parameters
    ----------
    train : Pandas DataFrame 
        Contain the training sample
    test : Pandas DataFrame 
        Contain the test(validation) sample
    inputs_features : list of str
        Contain the column name of the input features
    outputs_features : list of str
        Contain the column name of the features to be learned
    ylabel: list of str
        y-label        
    namefile_plot : str of the namefile to savethe figure. If empty (""), the figure is not saved
    """
    fig = plt.figure(figsize=(10.0, 3*len(outputs_features)))
    gs = GridSpec(len(outputs_features),5)

    # To get different color
    color=["rebeccapurple","red","royalblue","darkgoldenrod","green",'navy']
    for count,feature_to_learn in enumerate(outputs_features):
        
        # Make the second pannel with histogram and a gaussian fit
        ax2 = fig.add_subplot(gs[count,4])
        Nsigma_plot=10 # Number of stddev used to compute the precision of the ML
        stddev=np.std(test[inputs_features[count]]-test[feature_to_learn])
        Nbins=100
        Nhisto, bins=np.histogram(test[inputs_features[count]]-test[feature_to_learn],bins=np.linspace(-Nsigma_plot*stddev,Nsigma_plot*stddev,num=Nbins+1),density=False)
        Nhisto=Nhisto/Nhisto.max() # Normalize to have the max at 1

        bins_c=bins[:-1]+0.5*(bins[1]-bins[0]) # Get the center of the bins

        
        # Plot the histogram
        ax2.barh(bins_c,Nhisto,height=bins[1]-bins[0],color=color[count],alpha=0.6)

        ## Do the gaussian fitting
        gauss_model = models.Gaussian1D(amplitude=1.0, mean=0, stddev=stddev)
        fit_g = fitting.LevMarLSQFitter(); gauss = fit_g(gauss_model, bins_c, Nhisto)
        ampl=np.max(Nhisto)#gauss.amplitude.value
        mu=np.mean(test[inputs_features[count]]-test[feature_to_learn])#gauss.mean.value
        sigma=np.std(test[inputs_features[count]]-test[feature_to_learn])#gauss.stddev.value
        xplot=np.linspace(-Nsigma_plot*sigma,Nsigma_plot*sigma,1000)
        ax2.plot(gauss(xplot),xplot,'-',lw=2,color="k")
        ax2.axhline(y=0,color='k',lw=0.75,alpha=0.3)
        ax2.set_xlim(0,1.2)
        
        
        
        ## Get the limit for the yplots assuming a homemade receipts
        if(sigma>100):
            val_border_plot_y=math.ceil((5*sigma)/100)*100
            yticks=500
            val_ticks_y=math.ceil(val_border_plot_y/500.0)*500
        elif(sigma>10):
            val_border_plot_y=math.ceil((5*sigma)/100)*100
            yticks=250
            val_ticks_y=val_border_plot_y
        elif(sigma<0.55):
            val_border_plot_y=math.ceil((5*sigma)/0.5)*0.5
            yticks=0.5
            val_ticks_y=val_border_plot_y

        else:
            val_border_plot_y=math.ceil((5*sigma)/0.5)*0.5
            yticks=1.0
            val_ticks_y=val_border_plot_y

                    
        ## Same for the x value
        if((train[feature_to_learn].max()-train[feature_to_learn].min())//100>1):
            x_min_plot=math.floor(train[feature_to_learn].min()/500)*500
            x_max_plot=math.ceil(train[feature_to_learn].max()/500)*500
            xticks=250
        elif((train[feature_to_learn].max()-train[feature_to_learn].min())//2>1):
            x_min_plot=math.floor(train[feature_to_learn].min()/0.2)*0.2
            x_max_plot=math.ceil(train[feature_to_learn].max()/0.2)*0.2
            xticks=0.2
        else:
            x_min_plot=math.floor(train[feature_to_learn].min()/0.1)*0.1
            x_max_plot=math.ceil(train[feature_to_learn].max()/0.1)*0.1
            xticks=0.1
        if feature_to_learn=="MG_FE_APOGEE":
            x_min_plot=math.floor(train[feature_to_learn].min()/0.1)*0.1
            x_max_plot=math.ceil(train[feature_to_learn].max()/0.1)*0.1
            xticks=0.05
            
        ######################################################
        # Make first panel to see the residuals
        ax1 = fig.add_subplot(gs[count,0:4])
        
        # Get the uncertainties in different bins of the feature to learn
        features_means,bins_edges,_=stats.binned_statistic(train[feature_to_learn], train[inputs_features[count]]-train[feature_to_learn], statistic='mean',
                       bins=np.arange(x_min_plot-xticks/2,x_max_plot+xticks/2*1.001,xticks/2))
        features_stddev,_,_=stats.binned_statistic(train[feature_to_learn], train[inputs_features[count]]-train[feature_to_learn], statistic='std',
                       bins=bins_edges)
        features_count,_,_=stats.binned_statistic(train[feature_to_learn], train[inputs_features[count]]-train[feature_to_learn], statistic='count',
                       bins=bins_edges)
        
        # Put to nan bins with not enough data
        Ncut=5
        features_stddev=np.where(features_count>=Ncut,features_stddev,np.nan)
        features_means=np.where(features_count>=Ncut,features_means,np.nan)            
        bins_c=bins_edges#[:-1]+0.5*(bins_edges[1]-bins_edges[0])
        # Fill the values
        features_stddev=np.append(features_stddev,features_stddev[-1])
        features_means=np.append(features_means,features_means[-1])  
        features_count=np.append(features_count,features_count[-1])  

        #for i in range(0,len(bins_c)): # to Fill the values in the upper bin limit 
        #    if(features_count[i]==0 and features_count[i-1]>0):
        #        features_means[i]=features_means[i-1]
        #        features_stddev[i]=features_stddev[i-1]
        for i in range(0,len(bins_edges)-1):
            bins_c=np.array([bins_edges[i],bins_edges[i+1]])
            features_means_c=np.array([features_means[i],features_means[i]])
            features_stddev_c=np.array([features_stddev[i],features_stddev[i]])

            ax1.fill_between(bins_c,features_means_c-features_stddev_c,features_means_c+features_stddev_c,color=color[count],alpha=0.3)
            ax1.fill_between(bins_c,features_means_c-2*features_stddev_c,features_means_c+2*features_stddev_c,color=color[count],alpha=0.2)
            ax1.fill_between(bins_c,features_means_c-3*features_stddev_c,features_means_c+3*features_stddev_c,color=color[count],alpha=0.1)
        ax1.scatter(test[feature_to_learn],test[inputs_features[count]]-test[feature_to_learn],s=0.5,alpha=1.0,c="k",rasterized=True)        
        ax1.axhline(y=0,color='k',lw=0.75)
        ax1.axhline(y=-3*sigma,color='k',ls="--",lw=0.75)
        ax1.axhline(y=3*sigma,color='k',ls="--",lw=0.75)
        #ax1.axvline(x=train[feature_to_learn].min(),color='k',ls="--",lw=0.75)
        #ax1.axvline(x=train[feature_to_learn].max(),color='k',ls="--",lw=0.75)

        
        ax1.set_xlabel(feature_to_learn,fontsize=12)
        if ylabel is not None:
            ax1.set_ylabel(ylabel[count],fontsize=12)

        

        ##########################################################################
        ## Plot style    
        ax2.set_ylim(-val_border_plot_y-yticks/5,val_border_plot_y+yticks/5)
        ax1.set_ylim(-val_border_plot_y-yticks/5,val_border_plot_y+yticks/5)
        ax1.set_xlim(x_min_plot,x_max_plot)
        #ax1
        minor_locator = AutoMinorLocator(5)
        ax1.xaxis.set_minor_locator(minor_locator)
        ax1.tick_params(which='both', direction='in')
        minor_locatory = AutoMinorLocator(5)
        ax1.yaxis.set_minor_locator(minor_locatory)

        #ax2
        yticks2 = [0,0.5,1]
        tickLabels = map(str, yticks2)
        ax2.set_xticks(yticks2)
        ax2.set_xticklabels(tickLabels)
        ax2.xaxis.set_minor_locator(plt.MaxNLocator(13))
        ax2.yaxis.set_minor_locator(minor_locatory)

        ax2.yaxis.set_ticks_position("right") 
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.tick_params(which='both', direction='in')
        
        ax2.annotate(r"$\mu=$ %.3f"%mu+"\n"+"$\sigma$= %.3f" %(np.abs(sigma)), xy=(0.2,0.8), xycoords="axes fraction", color='black',size=10)



    fig.tight_layout()
    if(namefile_plot!=""):
        plt.savefig(namefile_plot, format='pdf')
    else:
        plt.show()
    plt.close()
    

    
    
##################################
## Generate analysis flag files ##
##################################

## To generate files where are stored the limit of the training sample to make quality flags 
def make_analysis_flags(inputs_features,outputs_features,train,Nbins=30,Nthres=1,suffix=""):
    """
    Description
    ----------
    Function to generate files that will store the boundary values of the training set then used to assert quality flags for the intrinsic network.

    Parameters
    ----------
    inputs_features : list of str
        Contain the column name of the input features used by the ANN
    outputs_features : list of str
        Contain the column name of the features to be learned
    train : Pandas DataFrame 
        Contain the training sample
    Nbins : int
        Number of bins to use to estimate the domain of validity for each parameters.
    Nthres : int
        Number of stars from the training sample that need to contain a bin to be consider as valid (bitmask=1).
    suffix : str
        Suffix to add to the name of the file where are store the quality flags 
    """

    ## Make the flags of the inputs features
    
    N2d_flag=int(np.sum(np.linspace(1,len(inputs_features)-1,len(inputs_features)-1))) # Number of 2d flag diagrams
    flags_matrix=np.zeros((N2d_flag,Nbins*Nbins,3))
    
    in_features_min_max=np.zeros((len(inputs_features),2))
    
    count=0
    for i,feature in enumerate(inputs_features):
        in_features_min_max[i,0]=np.min(train[feature])# Save the min value of the feature present in the training set
        in_features_min_max[i,1]=np.max(train[feature])# Save the max value of the feature present in the training set
        if(i<len(inputs_features)-1):
            for j,feature2 in enumerate(inputs_features[i+1:]):
                H, xedges, yedges = np.histogram2d(train[feature], train[feature2], bins=(Nbins, Nbins))
# Histogram does not follow Cartesian convention (see Notes),
# therefore transpose H for visualization purposes.
                H = H.T
                H=np.where(H>=Nthres,1,0) # Normalize the input to just get 0 or 1
                xval=xedges[:-1]+0.5*(xedges[1]-xedges[0])
                yval=yedges[:-1]+0.5*(yedges[1]-yedges[0])
                X, Y = np.meshgrid(xval,yval)
                out=np.zeros((Nbins*Nbins,3))
                out[:,0]=X.reshape((-1))
                out[:,1]=Y.reshape((-1))
                out[:,2]=H.reshape((-1))
                flags_matrix[count,:,0]=X.reshape((-1))
                flags_matrix[count,:,1]=Y.reshape((-1))
                flags_matrix[count,:,2]=H.reshape((-1))

                count+=1

    Table(flags_matrix).write("flag2d_inputs"+suffix+".fits",format="fits",overwrite=True)
    flag_1d=Table(in_features_min_max)
    flag_1d["Name"]=inputs_features
    flag_1d.write("flag1d_inputs"+suffix+".fits",format="fits",overwrite=True)

    
    ## Make the flags for the outputs features
    N2d_flag=int(np.sum(np.linspace(1,len(outputs_features)-1,len(outputs_features)-1))) # Number of 2d flag diagrams
    flags_matrix=np.zeros((N2d_flag,Nbins*Nbins,3))
    
    out_features_min_max=np.zeros((len(outputs_features),2))
    
    count=0
    for i,feature in enumerate(outputs_features):
        out_features_min_max[i,0]=np.min(train[feature])# Save the min value of the feature present in the training set
        out_features_min_max[i,1]=np.max(train[feature])# Save the max value of the feature present in the training set
        if(i<len(outputs_features)-1):
            for j,feature2 in enumerate(outputs_features[i+1:]):
                H, xedges, yedges = np.histogram2d(train[feature], train[feature2], bins=(Nbins, Nbins))
# Histogram does not follow Cartesian convention (see Notes),
# therefore transpose H for visualization purposes.
                H = H.T
                H=np.where(H>0,1,0) # Normalize the input to just get 0 or 1
                xval=xedges[:-1]+0.5*(xedges[1]-xedges[0])
                yval=yedges[:-1]+0.5*(yedges[1]-yedges[0])
                X, Y = np.meshgrid(xval,yval)
                out=np.zeros((Nbins*Nbins,3))
                out[:,0]=X.reshape((-1))
                out[:,1]=Y.reshape((-1))
                out[:,2]=H.reshape((-1))
                flags_matrix[count,:,0]=X.reshape((-1))
                flags_matrix[count,:,1]=Y.reshape((-1))
                flags_matrix[count,:,2]=H.reshape((-1))

                count+=1

    Table(flags_matrix).write("flag2d_outputs"+suffix+".fits",format="fits",overwrite=True)
    flag_1d=Table(out_features_min_max)
    flag_1d["Name"]=outputs_features
    flag_1d.write("flag1d_outputs"+suffix+".fits",format="fits",overwrite=True)


   
    
## Apply the Quality flags
def apply_analysis_flags(data,namefile_flag_suffix,inputs_features,outputs_features):
    """
    Description
    ----------
    Function to generate files that will store the boundary values of the training set then used to assert quality flags.

    Parameters
    ----------
    data : Pandas DataFrame 
        Contain the data sample
    namefile_flag_suffix : string
        Suffix given to the namefile where are store the flags.
   inputs_features : list of str
        Contain the column name of the input features used by the SpectroTranslator
    outputs_features : list of str
        Contain the column name of the features predicted by the SpectroTranslator      
        """
    
    ## Load the files
    flag2d_input=Table.read("flag2d_inputs"+namefile_flag_suffix+".fits")
    flag2d_output=Table.read("flag2d_outputs"+namefile_flag_suffix+".fits")
    
    
    sum_flag_in=np.ones(len(data))
    sum_flag_out=np.ones(len(data))
    
    if 'Qflag_comments' not in data.columns.values:
        data['Qflag_comments']=""

    # Quality flag on the inputs
    count=0
    for i,feature in enumerate(inputs_features):
        if(i<len(inputs_features)-1):
            for j,feature2 in enumerate(inputs_features[i+1:]):
                # Get the binned maps 
                flag=np.stack(flag2d_input.as_array()[count].tolist(), axis=0)
                Nbins=int(np.sqrt(flag.shape[0]))
                X=flag[:,0].reshape((Nbins,Nbins))[0,:]
                Y=flag[:,1].reshape((Nbins,Nbins)).T[0,:]
                H=flag[:,2].reshape((Nbins,Nbins))


                # Get the edges values of the 2 features
                Xedges=np.zeros(Nbins+1)
                Xbin_size=X[1]-X[0]
                Xedges[:-1]=X-0.5*Xbin_size
                Xedges[-1]=X[-1]+0.5*Xbin_size
                Yedges=np.zeros(Nbins+1)
                Ybin_size=Y[1]-Y[0]
                Yedges[:-1]=Y-0.5*Ybin_size
                Yedges[-1]=Y[-1]+0.5*Ybin_size

                pix_X=np.digitize(data[feature],Xedges)-1
                pix_Y=np.digitize(data[feature2],Yedges)-1
                pix_nb=pix_Y*(Nbins)+pix_X
                pix_nbtmp=np.where((pix_X>=0)&(pix_X<Nbins)&(pix_Y>=0)&(pix_Y<Nbins),pix_nb,0)
                flag_tmp=np.where((pix_X>=0)&(pix_X<Nbins)&(pix_Y>=0)&(pix_Y<Nbins),np.take(H,pix_nbtmp),0)
                sum_flag_in*=flag_tmp
                data['Qflag_comments']=np.where(flag_tmp==0,data['Qflag_comments']+"| Input "+feature+"_"+feature2+" outside training range",data['Qflag_comments']) ## Add the comments about the problem
                del(X,Y,H,flag,Xedges,Yedges,pix_nb,pix_nbtmp)

                count+=1
    data["Qflag_Input"]=np.where(sum_flag_in==1,True,False)
    del(sum_flag_in)


    # Quality flag on the outputs
    count=0
    for i,feature in enumerate(outputs_features):
        if(i<len(outputs_features)-1):
            for j,feature2 in enumerate(outputs_features[i+1:]):
                # Get the binned maps 
                flag=np.stack(flag2d_output.as_array()[count].tolist(), axis=0)
                Nbins=int(np.sqrt(flag.shape[0]))
                X=flag[:,0].reshape((Nbins,Nbins))[0,:]
                Y=flag[:,1].reshape((Nbins,Nbins)).T[0,:]
                H=flag[:,2].reshape((Nbins,Nbins))

                # Get the edges values of the 2 features
                Xedges=np.zeros(Nbins+1)
                Xbin_size=X[1]-X[0]
                Xedges[:-1]=X-0.5*Xbin_size
                Xedges[-1]=X[-1]+0.5*Xbin_size
                Yedges=np.zeros(Nbins+1)
                Ybin_size=Y[1]-Y[0]
                Yedges[:-1]=Y-0.5*Ybin_size
                Yedges[-1]=Y[-1]+0.5*Ybin_size

                pix_X=np.digitize(data[feature+"_50"],Xedges)-1
                pix_Y=np.digitize(data[feature2+"_50"],Yedges)-1
                pix_nb=pix_Y*(Nbins)+pix_X
                pix_nbtmp=np.where(((pix_X>=0)&(pix_X<Nbins)&(pix_Y>=0)&(pix_Y<Nbins)),pix_nb,0)
                flag_tmp=np.where((pix_X>=0)&(pix_X<Nbins)&(pix_Y>=0)&(pix_Y<Nbins),np.take(H,pix_nbtmp),0)
                sum_flag_out*=flag_tmp
                data['Qflag_comments']=np.where(flag_tmp==0,data['Qflag_comments']+"| Output "+feature+"_"+feature2+" outside training range",data['Qflag_comments']) ## Add the comments about the problem

                del(X,Y,H,flag,Xedges,Yedges,pix_nb,pix_nbtmp)

                count+=1
    data['Qflag_comments']=np.where(data['Qflag_comments']!="",data['Qflag_comments']+"|","")

    data["Qflag_Output"]=np.where(sum_flag_out==1,True,False)
    del(sum_flag_out)
