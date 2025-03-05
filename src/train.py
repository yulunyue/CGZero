from __future__ import absolute_import, division, print_function, unicode_literals
# Install TensorFlow

import tensorflow as tf
#THREADS=3
#tf.config.threading.set_intra_op_parallelism_threads(THREADS)
#tf.config.threading.set_inter_op_parallelism_threads(THREADS)
import random
import numpy as np
from numpy import genfromtxt
import pandas as pd
import base64
import sys
import glob, os
import re
import gc
import subprocess
import datetime
import queue

from tensorflow.python.client import device_lib

print("Python:"+sys.version)
print("TF:"+tf.__version__)
print("GPU:"+str(tf.test.is_gpu_available())+" CUDA:"+str(tf.test.is_built_with_cuda()))
#print(device_lib.list_local_devices())
import os
if not os.path.exists('/run/shm/traindata'):
    os.makedirs('/run/shm/traindata')
######################################### HYPERPARAMETERS AND CONFIGS #####################################

#Don't touch INPUT_SIZE UNLESS YOU ARE CHANGING THE NN INPUTS DEFINITION. POLICY_SIZE IS THE NUMBER OF LEGAL MOVES ON THE GAME.
#INPUT_SIZE must match what CGZero.cpp says in function:
# int  _Game::getInputDimensions() {...}
INPUT_SIZE=6*2*24+2*27
#The same than CGZero.cpp function: int getPolicySize() {...}
POLICY_SIZE=6


#HYPERPARAMETERS. You MUST assign them a value.
#DISCLAIMER: I have ABSOLUTELY no idea what are good hyperparameters. Most AZ documentation is confusing or even contradictory.
MATCHES_PER_GENERATION=400   #Something around 400 and 3000
PIT_MATCHES=400  #I have about 400, too little matches will not be enough to know if a candidate is best.
THREADS=6  #CPU dependant, on a Core i7-8700K I can set it to 9. With 6 I can use the PC for other purposes, like gaming.
TRAINING_POOL_SIZE=1009      #At least 500000, but can be millions.  500k-2M I'd say
                         #If it's too small there won't be enough samples.
                         #If it's too big it will use very old samples that might not have good values to learn.
TRAINING_SUBSET_SIZE=200   #I noticed better learning with more samples. 200k is not enough, 500k and up seemed a better value
K_BATCH_SIZE=64 #I have no idea what to use, I used 64 in last tests, but on others I put bigger batch sizes, like 512 or 1024
K_ITERATIONS=1 #Unused, this was for the minibatch approach from original Alphazero. There is commented code in the training loop about that
K_EPOCH=40  #Number of full learning passes the Tensorflow will do with the samples subset, from 1 to 60 I guess. If the learning process is slow I put a lower value.

K_WEIGHT_POLICY=1.0  #Give more importance to Policy losses
K_WEIGHT_VALUE=1.0   

WINRATE_ACCEPTED=55.6 #Or 55.0, what you prefer

K_LEARNING_RATE=0.001 #I imagine that at some point it must be lowered, I have no idea exactly when.
K_MOMENTUM=0.87 #Some hyperparameter for the training part. See https://distill.pub/2017/momentum/ 

#Parameters that controls how the endgame score is backpropagated to the samples.
#https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628
#For each sample the value I use is= B * ENDGAME_SCORE + (1.0 - B)*MeanScore
#Where B is  B = PROPAGATE_BASE + PROPAGATE_INC * (seenMoves / totalMovesInReplayBuffer)
#  and MeanScore is the MCTS mean value (Q).
#Alphazero uses z: PROPAGATE_BASE = 1.0 and PROPAGATE_INC = 0.0
#Others use q: PROPAGATE_BASE = 0.0 and PROPAGATE_INC = 0.0
#I'm using a mix of z and q, something like PROPAGATE_BASE = 0.45 and PROPAGATE_INC = 1.0-PROPAGATE_BASE
PROPAGATE_BASE=0.5   # 0.0 to 1.0. Percentage of endgame Score (-1 for loss and +1 for win) that the sample on turn 0 will have.
PROPAGATE_INC=1.0-PROPAGATE_BASE #Percentage of endgame score at end.

#I've used the same idea for the policy part. I dislike the "temperature" thing on Alphazero, I just tweak samples to be
# POLICY = B * VisitsPOLICY + (1.0 - B)*OneHotPOLICY
#VisitsPOLICY is calculated by dividing visitsChildren/visitsParent, so you have a 100% distribution
#OneHotPOLICY is all zeros except the selected move that it's a 1.0. It's also a 100% distribution
POLICY_BACKP_FIRST=10 #; //Similarly , but with percentage of turns, first 10% of turns doesn't have any "temperature",
POLICY_BACKP_LAST=5 #; //from (100-5=95%) I linearly sharpen policy to get only the best move, a one-hot policy

#TRAINING PARAMETERS
#I literally have no idea what's going on with the cpuct hyperparameter. It controls the exploration part on the MCTS search.
#But the problem is that I don't see any consensus about what's the best way to control it.
#https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/   Cpuct is not a constant!!!!!!!
#https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a  Trial and error way
TRAIN_CPUCT_MIN=2.5  #Maybe something between 2.0 and 3.0, but I'm clueless. 1.0 was "good" (>70% winrate) on pit, but not that good when submitted to CG
TRAIN_CPUCT_INC=0.00
TRAIN_CPUCT_MAX=TRAIN_CPUCT_MIN

TRAIN_NOISE_DIR_EPSILON=0.2 #it gives diversity, something between 0.10 and 0.30 can be good for selfplay
TRAIN_NOISE_DIR_ALPHA=1.3 #More than 1.0 always, maybe in the 1.0-1.6 range.
TRAIN_NOISE_DIR_DECAY=0.0 #Reduce noise each turn. Maybe it's not good to have much noise at endgame.
TRAIN_MCTS_ITER=800 #At least 800, but the more, the better quality of predictions but slower sample generation. 2k or 4k works fine too
TRAIN_NOISE_RANDOM=0.03 #Simplistic random noise to NN value. 0.03 means a  randomFloat(0.97,1.03)*NNvalue
#Don't touch that
TRAIN_PARAMS = f"{TRAIN_CPUCT_MIN} {TRAIN_CPUCT_INC} {TRAIN_CPUCT_MAX} {TRAIN_MCTS_ITER} {TRAIN_NOISE_DIR_EPSILON} {TRAIN_NOISE_DIR_ALPHA} {TRAIN_NOISE_DIR_DECAY} {TRAIN_NOISE_RANDOM} {PROPAGATE_BASE} {PROPAGATE_INC} {POLICY_BACKP_FIRST} {POLICY_BACKP_LAST}"

#PIT PARAMETERS
PIT_CPUCT_MIN=TRAIN_CPUCT_MIN
PIT_CPUCT_INC=0.00
PIT_CPUCT_MAX=PIT_CPUCT_MIN

PIT_NOISE_DIR_EPSILON=0.03 #Use much lower dirichlet noise than in selfplay. We need diversity but not noise.
PIT_NOISE_DIR_ALPHA=1.0
PIT_NOISE_DIR_DECAY=0.0006
PIT_MCTS_ITER=TRAIN_MCTS_ITER
PIT_NOISE_RANDOM=0.02
#Don't touch that
PIT_PARAM_THREAD= f"{THREADS} {PIT_MATCHES}"
PIT_PARAM_MCTS = f"{PIT_CPUCT_MIN} {PIT_CPUCT_INC} {PIT_CPUCT_MAX} {PIT_MCTS_ITER} {PIT_NOISE_DIR_EPSILON} {PIT_NOISE_DIR_ALPHA} {PIT_NOISE_DIR_DECAY} {PIT_NOISE_RANDOM} {PROPAGATE_BASE} {PROPAGATE_INC} {POLICY_BACKP_FIRST} {POLICY_BACKP_LAST}"
SAMPLES_FILE=os.path.join(".","traindata","samples.dat")
sampler_process=os.path.join(".","NNSampler")+" "+os.path.join(".","traindata")+" Replay.*.dat "+SAMPLES_FILE+" "+str(TRAINING_POOL_SIZE)+" "+str(TRAINING_SUBSET_SIZE)+" "+str(INPUT_SIZE)+" "+str(1+POLICY_SIZE)+" 1"

######################################### FUNCTIONS USED FOR TRAINING #####################################
def adjustLR():
    if generation == 0:
        return 8*K_LEARNING_RATE
    elif generation < 5:
        return 6*K_LEARNING_RATE
    elif generation < 10:
        return 2*K_LEARNING_RATE
    else:
        return K_LEARNING_RATE

#Save weights, non trainable layers must be named "IGNORE_*"
def SaveModel(my_model,fileSTR):
    totalbytes=0
    data=[]
    Wmodel = open("./"+fileSTR, "wb")
    for x in my_model.weights:
        if ("IGNORE_" in x.name):
            #print("Ignoring layer "+x.name)
            continue
        nn = x.numpy()
        T = nn
        v = np.ndarray.tobytes(T)
        Wmodel.write(bytearray(v))
        totalbytes+=len(v)
        data.append(base64.b64encode(v).decode("utf-8"))
    Wmodel.close()
def readWinrate(candidatefile,bestfile):
    files=sorted(glob.glob(os.path.join('pitresults','Pit_'+candidatefile+'_'+bestfile+'_*.txt')),reverse=True)
    for file in files:
        with open(file, 'r') as f:
            strvalue=f.read().strip()
        return float(strvalue)
    return -1.0
def readAndDeleteWinrate(candidatefile,bestfile):
    files=sorted(glob.glob(os.path.join('pitresults','Pit_'+candidatefile+'_'+bestfile+'_*.txt')),reverse=True)
    valor=-1.0
    for file in files:
        if (valor == -1.0):
            with open(file, 'r') as f:
                valor=float(f.read().strip())
        os.remove(file)
    return valor

#LR_decay = tf.keras.callbacks.LearningRateScheduler(tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=20000,decay_rate=0.9))

#It's INCOMPLETE, YOU MUST CREATE YOUR OWN MODEL!!
#Order is important! You must take into account that this model and C++ counterpart must be synchronized, so layer orders must be the same between languages.

#Input layer
inputs =  tf.keras.Input(shape=(INPUT_SIZE,), name='input')
#Common part of the Model, both policy and value use these layers
x = tf.keras.layers.Dense(TODOTODOTODOT,activation='relu',name='Dense1')(inputs)
#x = tf.keras.layers.Dense(TODOTODOTODOT,activation='relu')(x)

#Split part 1, P1 layers are for the policy part. If you don't want extra layers for p1, I guess you can do p1=x
p1 = tf.keras.layers.Dense(TODOTODOTODOT,activation='relu',name='p1')(x)
#p1 = tf.keras.layers.Dense(TODOTODOTODOT,activation='relu')(p1)

#Split part, v1 layers are for the value part. If you don't want extra layers for v1, I guess you can do v1=x
v1 = tf.keras.layers.Dense(TODOTODOTODOT,activation='relu',name='v1')(x)
#v1 = tf.keras.layers.Dense(TODOTODOTODOT,activation='relu')(v1)
#Output layers, don't touch them if you don't know what are you doing.
value = tf.keras.layers.Dense(1, activation='tanh',name='value')(v1)
policy = tf.keras.layers.Dense(POLICY_SIZE, activation='softmax',name='policy')(p1)
model = tf.keras.Model(inputs=inputs, outputs=[value, policy])
#Others use Adam as optimizer, I just used SGD because I'm clueless and I saw some others using SGD.
opt = tf.keras.optimizers.SGD(learning_rate=K_LEARNING_RATE, momentum=K_MOMENTUM)
#Keep a file with losses history
csv_logger = tf.keras.callbacks.CSVLogger('gen_train.log',append=True)

model.compile(loss={'value': 'mean_squared_error',
                    'policy':tf.keras.losses.KLD  },
              optimizer=opt,
              loss_weights = {'value':K_WEIGHT_VALUE,
                              'policy':K_WEIGHT_POLICY}
              ,metrics={'value':'mean_absolute_percentage_error',
                       'policy': 'categorical_accuracy' }
             )

if not os.path.exists('gen0000.h5'):
    model.save('gen0000.h5')
    SaveModel(model,"gen0000.w32")
    model.summary()
generation=0
gen_name="gen"+str(generation).zfill(4)
if os.path.exists('generation.txt'):
    with open('generation.txt', 'r') as f:
        generation = int(f.read().strip())
        gen_name="gen"+str(generation).zfill(4)
        print("Generation is :"+gen_name+" "+str(generation))        
        if (generation > 0):
            #model =  tf.keras.models.load_model(gen_name+'.h5', custom_objects={"policy_loss": policy_loss})
            model =  tf.keras.models.load_model(gen_name+'.h5')
if not os.path.exists(gen_name+".w32"):
    SaveModel(model,gen_name+".w32")
    model.save(gen_name+'.h5')
gen_best1=gen_name
if os.path.exists('gen_best1.txt'):
    with open('gen_best1.txt', 'r') as f:
        gen_best1 = f.read().strip()
gen_best2=gen_name
if os.path.exists('gen_best2.txt'):
    with open('gen_best2.txt', 'r') as f:
        gen_best2 = f.read().strip()        
model=tf.keras.models.load_model(gen_name+'.h5')
print("Best Model1:"+gen_best1+" + "+gen_best2+". Current generation:"+gen_name+" loaded")

pit_winrate2=70.0
pit_winrate=max(0,100.0-pit_winrate2)
    
while True:
    model.optimizer.learning_rate.assign(adjustLR())
    samplescount=0
    #if (generation == 0):
     #   csv_data = np.fromfile(SAMPLES_FILE, dtype=np.float32)
      #  csv_data=np.reshape(csv_data, (-1,INPUT_SIZE+POLICY_SIZE+2))
       # samplescount =(csv_data.shape)[0]

    #Remove generation 0 data once we have samples with more quality
    if (generation > 4 and generation < 6):
        gen0="gen"+str(0).zfill(4)
        remove0=glob.glob(os.path.join('traindata','Replay_'+'*'+gen0+"*"+'.dat'))
        for filePath in remove0:
            try:
                os.remove(filePath)
            except:
                pass
    # 1- Creates self plays between best models (and also some random generation to give diversity)
    # The code will repeat until it has enough samples (only after generation 10, lower ones are of bad quality).
    while samplescount < TRAINING_SUBSET_SIZE/3:
        if (generation<=1):
            random_enemy=gen_name
        else:
            random_enemy="gen"+str(random.randint(max(1,generation-5), generation)).zfill(4)
        
        if (generation==0 and (len(glob.glob(os.path.join('traindata','Replay_'+'*'+gen_name+"vs"+gen_name+'.dat')))>0)):
            print('Replay_'+'*'+gen_name+'.txt already exists') 
        else:
            pFirst=max(0.2,0.9*(1.0-(pit_winrate/(pit_winrate+pit_winrate2))))
            pSecond=0.9-pFirst
            print(f" **** Doing samples. Count:{samplescount}. pBest1:{100.0*pFirst}% p2:{100.0*pSecond}%  {pit_winrate} {pit_winrate2}")
            p70=int(pFirst*MATCHES_PER_GENERATION)
            p20=int(pSecond*MATCHES_PER_GENERATION)
            p5=MATCHES_PER_GENERATION-p70-p20
            selfplay_process=os.path.join(".","CGZero")+" selfplay "+f"{THREADS} {p70} "+gen_best1+" "+TRAIN_PARAMS+" "+gen_best1+" "+TRAIN_PARAMS
            print(selfplay_process)
            subprocess.run(selfplay_process, shell=True)
            if (gen_best1 != gen_best2):
                selfplay_process=os.path.join(".","CGZero")+" selfplay "+f"{THREADS} {p20} "+gen_best1+" "+TRAIN_PARAMS+" "+gen_best2+" "+TRAIN_PARAMS
                print(selfplay_process)
                subprocess.run(selfplay_process, shell=True)
                if (p70 >= p20):
                    A=gen_best1 if (gen_best1 >= random_enemy) else random_enemy
                    B=gen_best1 if (A == random_enemy) else random_enemy
                else:
                    A=gen_best2 if (gen_best2 >= random_enemy) else random_enemy
                    B=gen_best2 if (A == random_enemy) else random_enemy
                selfplay_process=os.path.join(".","CGZero")+" selfplay "+f"{THREADS} {p5} "+A+" "+TRAIN_PARAMS+" "+B+" "+TRAIN_PARAMS
                print(selfplay_process)
                subprocess.run(selfplay_process, shell=True)
        print('Reading training data')
        # 2- Creates a random sample dataset from selfplay
        print(sampler_process)
        subprocess.run(sampler_process, shell=True)
        csv_data = np.fromfile(SAMPLES_FILE, dtype=np.float32)
        csv_data=np.reshape(csv_data, (-1,INPUT_SIZE+POLICY_SIZE+2))
        gc.collect()
        samplescount =(csv_data.shape)[0]
        if (generation < 10):
            break
    #Cut samples to inputs , policy , value , countVisits
    np.random.shuffle(csv_data)
    cut_index = [(csv_data.shape)[1]-POLICY_SIZE-2, (csv_data.shape)[1]-2,(csv_data.shape)[1]-1]
    samples,policy,value,countVisits=np.split(csv_data, cut_index,axis=1)
    
    #mask= np.where(policy < 0, -999999999.99, policy) #not used, it was a test to remove invalid moves, not needed
    policy= np.where(policy < 0, 0, policy) 
    csv_data=None
    gc.collect()
    # 3- Train the current model
    ####Minibatches learning: This is for doing minibatches, but I prefer to simply feed all the samples subset.
    #for loop in range(K_ITERATIONS):
    #    print("Batch "+str(loop)+":",end='')
    #    indices = np.random.choice(value.shape[0], K_BATCH_SIZE, replace=False)
    #    S=samples[indices]
    #    P=policy[indices]
    #    V=value[indices]
    #    #model.optimizer.learning_rate.assign(learning_rate_scheduler(loop, 0.0))
    #    #model.optimizer.learning_rate.assign(K_LEARNING_RATE)
    #    model.fit({'input':S}, {'policy': P, 'value':V},verbose=2, epochs=K_EPOCH,callbacks=[csv_logger],batch_size=int(K_BATCH_SIZE/4))
    
    ####Simple learning, just learn from all the subset. If you activate the former, disable this line
    model.fit({'input':samples}, {'policy': policy, 'value':value},verbose=2, epochs=min(generation+1,K_EPOCH),callbacks=[csv_logger],batch_size=int(K_BATCH_SIZE))
    gc.collect()
    #new generation
    print('New generation '+gen_name+' -> '+"gen"+str(generation+1).zfill(4))
    generation=generation+1
    gen_name="gen"+str(generation).zfill(4)
    # 4- Save candidate as a new generation (useful to resume later, or return back to a previous train state)
    print('Save Model '+gen_name+'.h5')
    model.save(gen_name+'.h5')
    SaveModel(model,gen_name+".w32")
    with open('generation.txt', 'w') as f:
        f.write(str(generation))    
        
    # 5- Pit play vs best1 and best2, promote candidate as best if winrate is good
    pitplay_process=os.path.join(".","CGZero")+" pitplay "+PIT_PARAM_THREAD+" "+gen_name+" "+PIT_PARAM_MCTS+" "+gen_best1+" "+PIT_PARAM_MCTS
    print('subprocess.run('+pitplay_process+', shell=True)')
    subprocess.run(pitplay_process, shell=True)
    pit_winrate=readWinrate(gen_name,gen_best1)
    if gen_best1 == gen_best2:
        pit_winrate2=pit_winrate
    else:
        pitplay_process=os.path.join(".","CGZero")+" pitplay "+PIT_PARAM_THREAD+" "+gen_name+" "+PIT_PARAM_MCTS+" "+gen_best2+" "+PIT_PARAM_MCTS
        subprocess.run(pitplay_process, shell=True)
        pit_winrate2=readWinrate(gen_name,gen_best2)
    print('Winrate '+str(pit_winrate)+' '+str(pit_winrate2))
    #Check if it's a new best, update bests
    if (pit_winrate>=WINRATE_ACCEPTED):
        print("New best:"+gen_name+" vs "+gen_best1+": Winrate1:"+str(pit_winrate)+"%")
        print("        :"+gen_name+" vs "+gen_best2+": Winrate2:"+str(pit_winrate2)+"%")
        tmpgenbest1=gen_best1
        gen_best1=gen_name
        with open('gen_best1.txt', 'w') as f:
            f.write(gen_best1)
        if (pit_winrate2>=50.0 and tmpgenbest1 != gen_best2):
            gen_best2=tmpgenbest1
            with open('gen_best2.txt', 'w') as f:
                f.write(gen_best2)