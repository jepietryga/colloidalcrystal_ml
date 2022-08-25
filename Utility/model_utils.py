# Series of helper functions for reading data
import numpy as np
def categorical_data_translator(passed_list):
    '''
    This is hard-coded since we know our own classifications
    '''

    num_list = []
    index_track = 0
    for item in passed_list:
        if item == 'Crystal':
            num_list.append(2)
        elif item == 'Multiple Crystal':
            num_list.append(3)
        elif item == 'Poorly Segmented':
            num_list.append(1)
        elif item == 'Incomplete':
            num_list.append(0)
        else:
            print(f'Error: {item} is unknown ID at {index_track}')
            exit()
        index_track = index_track + 1
    
    return num_list


def success_of_guess(y_pred,y_test,ohe):
    '''
    Given the predicted results, for each label, how does our model perform?

    Args:
    y_pred (array) : Our predicted values in OneHotEncoding
    y_test (array) : Our test values in OneHotEncoding
    ohe (OneHotEncoder) : Our OneHotEncoder (for translating meaning)

    Returns:
    success (ndarray) : Successful guesses for each label
    failed_to_guess (ndarry) : Number of a times a label should've been guessed, but was missed
    incorrectly_guessed (ndarry) : Number of times a label was mistakenly guessed
    paired_guess (ndarry) : 2D Array where -1 indicates the guess and 1 indicates the correct answer. 
                            All 0s implies a successful guess
    '''
    success = np.zeros([np.size(y_pred[0]),])
    failed_to_guess = np.zeros([np.size(y_pred[0]),])
    incorrectly_guessed = np.zeros([np.size(y_pred[0]),])
    paired_guess = y_test-y_pred
    for ii in np.arange(np.shape(y_pred)[0]):
        if np.where(y_pred[ii] == 1) == np.where(y_test[ii] == 1):
            success += y_pred[ii]
        else:
            incorrectly_guessed += y_pred[ii]
            failed_to_guess += y_test[ii]
    
    labels_list = ohe.get_feature_names_out()

    for ii in np.arange(np.size(labels_list)):
        recall = success[ii]/(success[ii]+failed_to_guess[ii])
        precision = success[ii]/(success[ii]+incorrectly_guessed[ii])
        f1 = 2*precision*recall/(precision+recall)
        print(f'{labels_list[ii]} -> Precision = {precision}, Recall = {recall}, F1 = {f1}')
    accuracy = np.sum(success)/(np.shape(y_pred)[0])
    print(f'Run Accuracy : {accuracy}')