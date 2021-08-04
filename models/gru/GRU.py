import pickle
import argparse
import shutil
import os
import random

import tensorflow as tf
import numpy as np

global ARGS

def getNumberOfCodes(sets):
  highestCode = 0
  for set in sets:
    for pat in set:
      for adm in pat:
        for code in adm:
          if code > highestCode:
            highestCode = code
  return (highestCode + 1)


def prepareHotVectors(train_tensor, labels_tensor):
  nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_tensor]) - 1
  numberOfPatients = len(train_tensor)
  maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

  x_hotvectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float64)
  y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float64)
  mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(np.float64)

  for idx, (train_patient_matrix,label_patient_matrix) in enumerate(zip(train_tensor,labels_tensor)):
    for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
      for code in visit_line:
        x_hotvectors_tensorf[i_th_visit, idx, code] = 1
    for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
      for code in visit_line:
        y_hotvectors_tensor[i_th_visit, idx, code] = 1
    mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

  nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=np.int32)
  return x_hotvectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List


def load_data():
  main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".train dimensions ")
  main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
  print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".test dimensions ")
  print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")

  ARGS.numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
  print('Number of diagnosis input codes: ' + str(ARGS.numberOfInputCodes))

  #uses the same data for testing, but disregarding the fist admission of each patient
  labels_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  labels_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))

  train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
  main_trainSet = [main_trainSet[i] for i in train_sorted_index]
  labels_trainSet = [labels_trainSet[i] for i in train_sorted_index]

  test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
  main_testSet = [main_testSet[i] for i in test_sorted_index]
  labels_testSet = [labels_testSet[i] for i in test_sorted_index]

  trainSet = [main_trainSet, labels_trainSet]
  testSet = [main_testSet, labels_testSet]

  return trainSet, testSet

def performEvaluation(session, loss, x, y, mask, seqLen, test_Set):
  batchSize = ARGS.batchSize

  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
  crossEntropySum = 0.0
  dataCount = 0.0
  #computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
  with session.as_default() as sess:
    for index in range(n_batches):
      batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
      batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
      xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)

      feed_dict = {x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List}
      crossEntropy = sess.run(loss, feed_dict=feed_dict)

      #accumulation by simple summation taking the batch size into account
      crossEntropySum += crossEntropy * len(batchX)
      dataCount += float(len(batchX))
      #At the end, it returns the mean cross entropy considering all the batches
  return n_batches, crossEntropySum / dataCount

def GRU_layer(inputTensor, seqLen):
  grus = [tf.nn.rnn_cell.GRUCell(size) for size in ARGS.hiddenDimSize] #According to docs (https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell), the peephole version is based on LSTM Google (2014)
  drops = [tf.nn.rnn_cell.DropoutWrapper(gru, state_keep_prob=(1-ARGS.dropoutRate)) for gru in grus]
  cell = tf.nn.rnn_cell.MultiRNNCell(drops)
  gru_outputs, gru_states = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=seqLen, time_major=True, dtype=tf.float32)

  return gru_states[-1]

def FC_layer(inputTensor):
  im_dim = inputTensor.get_shape()[-1]
  weights = tf.get_variable(name='weights',
                           shape=[im_dim, ARGS.numberOfInputCodes],
                           dtype=tf.float32,
                           initializer=tf.keras.initializers.glorot_normal())

  bias = tf.get_variable(name='bias',
                       shape=[ARGS.numberOfInputCodes],
                       dtype=tf.float32,
                       initializer=tf.zeros_initializer())
  output = tf.nn.softmax(tf.nn.relu(tf.add(tf.matmul(inputTensor, weights), bias)))
  return output, weights


def build_model():
  graph = tf.Graph()
  with graph.as_default():
    xf = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes], name="inputs")
    yf = tf.placeholder(tf.float32, [None, None, ARGS.numberOfInputCodes], name="labels")
    maskf = tf.placeholder(tf.float32, [None, None], name="mask")
    seqLen = tf.placeholder(tf.float32, [None], name="nVisitsOfEachPatient_List")

    with tf.device('/gpu:0'):
      flowingTensor = GRU_layer(xf, seqLen)
      flowingTensor, weights = FC_layer(flowingTensor)
      flowingTensor = tf.math.multiply(flowingTensor, maskf[:,:,None], name="predictions")

      epislon = 1e-8
      cross_entropy = -(yf * tf.log(flowingTensor + epislon) + (1. - yf) * tf.log(1. - flowingTensor + epislon))
      prediction_loss = tf.math.reduce_mean(tf.math.reduce_sum(cross_entropy, axis=[2, 0]) / seqLen)
      L2_regularized_loss = prediction_loss + tf.math.reduce_sum(ARGS.LregularizationAlpha * (weights ** 2))

      optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-06).minimize(L2_regularized_loss)
    return tf.global_variables_initializer(), graph, optimizer, L2_regularized_loss, xf, yf, maskf, seqLen, flowingTensor

def train_model():
  print("==> data loading")
  trainSet, testSet = load_data()

  print("==> model building")
  init, graph, optimizer, loss, x, y, mask, seqLen, predictions = build_model()

  print ("==> training and validation")
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))

  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelDirName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0

  with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)

    for epoch_counter in range(ARGS.nEpochs):
      iteration = 0
      trainCrossEntropyVector = []
      for index in random.sample(range(n_batches), n_batches):
        batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
        batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
        xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
        xf += np.random.normal(0, 0.1, xf.shape)

        feed_dict = {x: xf, y: yf, mask: maskf, seqLen: nVisitsOfEachPatient_List}
        _, trainCrossEntropy = sess.run([optimizer, loss], feed_dict=feed_dict)

        trainCrossEntropyVector.append(trainCrossEntropy)
        iteration += 1

      print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
      nValidBatches, validationCrossEntropy = performEvaluation(sess, loss, x, y, mask, seqLen, testSet)
      print('      mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))

      if validationCrossEntropy < bestValidationCrossEntropy:
        iImprovementEpochs += 1
        iConsecutiveNonImprovements = 0
        bestValidationCrossEntropy = validationCrossEntropy
        bestValidationEpoch = epoch_counter

        if os.path.exists(bestModelDirName):
          shutil.rmtree(bestModelDirName)
        bestModelDirName = ARGS.outFile + '.' + str(epoch_counter)

        if os.path.exists(bestModelDirName):
          shutil.rmtree(bestModelDirName)

        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs= {"inputs": x, "mask": mask, "seqLen": seqLen}, outputs= {"predictions": predictions})
        builder = tf.saved_model.builder.SavedModelBuilder(bestModelDirName)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'model': signature})
        builder.save()

      else:
        print('Epoch ended without improvement.')
        iConsecutiveNonImprovements += 1
      if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
        break

    # Best results
    print('--------------SUMMARY--------------')
    print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (
    bestValidationEpoch, bestValidationCrossEntropy))
    print('Best model file: ' + bestModelDirName)
    print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter + 1) + ' possible improvements.')
    print('Note: the smaller the cross entropy, the better.')
    print('-----------------------------------')
    sess.close()


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
  parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file directory to store the model.')
  parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=5, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training')
  parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
  parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
  parser.add_argument('--dropoutRate', type=float, default=0.45, help='Dropout probability.')

  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp

if __name__ == '__main__':
  global ARGS
  ARGS = parse_arguments()

  train_model()