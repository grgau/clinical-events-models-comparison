import pickle
import argparse
import os
import random
import tensorflow as tf
import numpy as np
from sklearn import metrics

global ARGS

def prepareHotVectors(test_tensor, labels_tensor):
  n_visits_of_each_patientList = np.array([len(seq) for seq in test_tensor]) - 1
  number_of_patients = len(test_tensor)
  max_number_of_visits = np.max(n_visits_of_each_patientList)

  x_hotvectors_tensorf = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(np.float64)
  y_hotvectors_tensor = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(np.float64)

  mask = np.zeros((max_number_of_visits, number_of_patients)).astype(np.float64)

  for idx, (test_patient_matrix,label_patient_matrix) in enumerate(zip(test_tensor,labels_tensor)):
    for i_th_visit, visit_line in enumerate(test_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
      for code in visit_line:
        x_hotvectors_tensorf[i_th_visit, idx, code] = 1
    for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
      for code in visit_line:
        y_hotvectors_tensor[i_th_visit, idx, code] = 1
    mask[:n_visits_of_each_patientList[idx], idx] = 1.

  return x_hotvectors_tensorf, y_hotvectors_tensor, mask, n_visits_of_each_patientList

def loadModel():
  model_path = ARGS.modelPath

  loaded_graph = tf.Graph()
  with tf.Session(graph=loaded_graph, config=tf.ConfigProto(allow_soft_placement=True)).as_default() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    x = loaded_graph.get_tensor_by_name('inputs:0')
    predictions = loaded_graph.get_tensor_by_name('predictions:0')
    mask = loaded_graph.get_tensor_by_name('mask:0')
    seqLen = loaded_graph.get_tensor_by_name('nVisitsOfEachPatient_List:0')

    ARGS.numberOfInputCodes = x.get_shape()[-1]
    return sess, predictions, x, mask, seqLen


def load_data():
  testSet_x = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
  testSet_y = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))

  def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

  sorted_index = len_argsort(testSet_x)
  testSet_x = [testSet_x[i] for i in sorted_index]
  testSet_y = [testSet_y[i] for i in sorted_index]

  testSet = [testSet_x, testSet_y]
  return testSet

def testModel():
  print('==> model loading')
  session, predictions, x, mask, seqLen = loadModel()

  print('==> data loading')
  testSet = load_data()

  # print ('==> load patients')
  # patientsSet = load_patients()
  # patientsSet = None

  print('==> model execution')
  nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
  predictedY_list = []
  predictedProbabilities_list = []
  actualY_list = []
  predicted_yList = []

  file = open(ARGS.inputFileRadical + 'AUCROC.input.txt', 'w')

  with session as sess:
    for batchIndex in range(nBatches):
      batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
      batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
      xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
      # retrieve the maximum number of admissions considering all the patients
      maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)
      # make prediction
      predicted_y = sess.run(predictions, feed_dict={x: xf, mask: maskf, seqLen: nVisitsOfEachPatient_List})
      predicted_yList.append(predicted_y.tolist()[-1])

      # traverse the predicted results, once for each patient in the batch
      for ith_patient in range(predicted_y.shape[1]):
        predictedPatientSlice = predicted_y[:, ith_patient, :]
        # retrieve actual y from batch tensor -> actual codes, not the hotvector
        actual_y = batchY[ith_patient][1:]
        # for each admission of the ith-patient
        for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
          # convert array of actual answers to list
          actualY_list.append(actual_y[ith_admission])
          # retrieves ith-admission of ths ith-patient
          ithPrediction = predictedPatientSlice[ith_admission]
          # since ithPrediction is a vector of probabilties with the same dimensionality of the hotvectors
          # enumerate is enough to retrieve the original codes
          enumeratedPrediction = [codeProbability_pair for codeProbability_pair in enumerate(ithPrediction)]
          # sort everything
          sortedPredictionsAll = sorted(enumeratedPrediction, key=lambda x: x[1], reverse=True)
          # creates trimmed list up to max(maxNumberOfAdmissions,30) elements
          sortedTopPredictions = sortedPredictionsAll[0:max(maxNumberOfAdmissions, 30)]
          # here we simply toss off the probability and keep only the sorted codes
          sortedTopPredictions_indexes = [codeProbability_pair[0] for codeProbability_pair in sortedTopPredictions]
          # stores results in a list of lists - after processing all batches, predictedY_list stores all the prediction results
          predictedY_list.append(sortedTopPredictions_indexes)
          predictedProbabilities_list.append(sortedPredictionsAll)

    # ---------------------------------Report results using k=[10,20,30]
    print('==> computation of prediction results with constant k')
    recall_sum = [0.0, 0.0, 0.0]

    k_list = [10, 20, 30]
    for ith_admission in range(len(predictedY_list)):
      ithActualYSet = set(actualY_list[ith_admission])
      for ithK, k in enumerate(k_list):
        ithPredictedY = set(predictedY_list[ith_admission][:k])
        intersection_set = ithActualYSet.intersection(ithPredictedY)
        recall_sum[ithK] += len(intersection_set) / float(
          len(ithActualYSet))  # this is recall because the numerator is len(ithActualYSet)

    precision_sum = [0.0, 0.0, 0.0]
    k_listForPrecision = [1, 2, 3]
    for ith_admission in range(len(predictedY_list)):
      ithActualYSet = set(actualY_list[ith_admission])
      for ithK, k in enumerate(k_listForPrecision):
        ithPredictedY = set(predictedY_list[ith_admission][:k])
        intersection_set = ithActualYSet.intersection(ithPredictedY)
        precision_sum[ithK] += len(intersection_set) / float(
          k)  # this is precision because the numerator is k \in [10,20,30]

    finalRecalls = []
    finalPrecisions = []
    for ithK, k in enumerate(k_list):
      finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))
      finalPrecisions.append(precision_sum[ithK] / float(len(predictedY_list)))

    print('Results for Recall@' + str(k_list))
    print(str(finalRecalls[0]))
    print(str(finalRecalls[1]))
    print(str(finalRecalls[2]))

    print('Results for Precision@' + str(k_listForPrecision))
    print(str(finalPrecisions[0]))
    print(str(finalPrecisions[1]))
    print(str(finalPrecisions[2]))

    # ---------------------------------Report results using k=lenght of actual answer vector
    print('==> computation of prediction results with dynamic k=lenght of actual answer vector times [1,2,3]')
    recall_sum = [0.0, 0.0, 0.0]
    precision_sum = [0.0, 0.0, 0.0]
    multiples_list = [0, 1, 2]
    for ith_admission in range(len(predictedY_list)):
      ithActualYSet = set(actualY_list[ith_admission])
        # print('--->Admission: ' + str(ith_admission))
      for m in multiples_list:
        k = len(ithActualYSet) * (m + 1)
        # print('K: ' + str(k))
        ithPredictedY = set(predictedY_list[ith_admission][:k])
        # print('Prediction: ' + str(ithPredictedY))
        # print('Actual: ' + str(ithActualYSet))
        intersection_set = ithActualYSet.intersection(ithPredictedY)
        # print('Intersection: ' + str(intersection_set))
        recall_sum[m] += len(intersection_set) / float(len(ithActualYSet))
        precision_sum[m] += len(intersection_set) / float(k)  # this is precision because the numerator is ithK \in [10,20,30]

    bReportDynamicK = False
    if bReportDynamicK:
      finalRecalls = []
      finalPrecisions = []
      for m in multiples_list:
        finalRecalls.append(recall_sum[m] / float(len(predictedY_list)))
        finalPrecisions.append(precision_sum[m] / float(len(predictedY_list)))

      print('Results for Recall@k*1, Recall@k*2, and Recall@k*3')
      print(str(finalRecalls[0]))
      print(str(finalRecalls[1]))
      print(str(finalRecalls[2]))

      print('Results for Precision@k*1, Precision@k*2, and Precision@k*3')
      print(str(finalPrecisions[0]))
      print(str(finalPrecisions[1]))
      print(str(finalPrecisions[2]))

    # ---------------------------------Write data for AUC-ROC computation
    bWriteDataForAUC = False
    fullListOfTrueYOutcomeForAUCROCAndPR_list = []
    fullListOfPredictedYProbsForAUCROC_list = []
    fullListOfPredictedYForPrecisionRecall_list = []
    for ith_admission in range(len(predictedY_list)):
      ithActualY = actualY_list[ith_admission]
      nActualCodes = len(ithActualY)
      ithPredictedProbabilities = predictedProbabilities_list[ith_admission]  # [0:nActualCodes]
      ithPrediction = 0
      for predicted_code, predicted_prob in ithPredictedProbabilities:
        fullListOfPredictedYProbsForAUCROC_list.append(predicted_prob)
        # for precision-recall purposes, the nActual first codes correspond to what was estimated as correct answers
        if ithPrediction < nActualCodes:
          fullListOfPredictedYForPrecisionRecall_list.append(1)
        else:
          fullListOfPredictedYForPrecisionRecall_list.append(0)

        # the list fullListOfTrueYOutcomeForAUCROCAndPR_list corresponds to the true answer, either positive or negative
        # it is used for both Precision Recall and for AUCROC
        if predicted_code in ithActualY:
          fullListOfTrueYOutcomeForAUCROCAndPR_list.append(1)
          file.write("1 " + str(predicted_prob) + '\n')
        else:
          fullListOfTrueYOutcomeForAUCROCAndPR_list.append(0)
          file.write("0 " + str(predicted_prob) + '\n')
        ithPrediction += 1
    file.close()

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    print("Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                                 fullListOfPredictedYProbsForAUCROC_list,
                                                                 average='weighted')))
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                        fullListOfPredictedYForPrecisionRecall_list,
                                                        average='binary')
    print('Precision: ' + str(PRResults[0]))
    print('Recall: ' + str(PRResults[1]))
    print('Binary F1 Score: ' + str(PRResults[2]))  # FBeta score with beta = 1.0
    print('Support: ' + str(PRResults[3]))

  sess.close()

  return predicted_yList

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
  parser.add_argument('modelPath', type=str, metavar='<model_path>', help='The path to the model directory')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--learningRate', type=float, default=0.5, help='Learning rate.')
  parser.add_argument('--dropoutRate', type=float, default=0.45, help='Dropout probability.')
  ARGStemp = parser.parse_args()
  ARGStemp.hiddenDimSizeStr = ARGStemp.hiddenDimSize
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp


if __name__ == '__main__':
  global ARGS
  ARGS = parse_arguments()

  predictions = testModel()