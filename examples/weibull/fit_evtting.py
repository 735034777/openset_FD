"""
File: fit_evtting
Author: admin
Date Created: 2024/7/16
Last Modified: 2024/7/16

Description:
    Describe the file purpose and main functionality.


"""
import sys
import numpy as np

fitting_size ,sign,scores_to_drop  =0,1,0

def FitHigh(inputData,inputDataSize,fsize):
    global fitting_size,sign
    if fsize>0:
        fitting_size=fsize
    sign = 1
    return EvtGeneric(inputData, inputDataSize)

def EvtGeneric(inputData, inputDataSize, inward, x):
    global fitting_size,sign
    if fitting_size > inputDataSize:
        print(
            f"In MetaRecognition, warning asked to fit with tail size {fitting_size} but input data only {inputDataSize}, returning -1",
            file=sys.stderr)
        return -1
    inputData = np.array(inputData, dtype=np.double)
    inputDataCopy = np.copy(inputData)
    dataPtr = None
    icnt = 0

    if not inward and sign > 0:
        icnt = inputDataSize
        inputDataCopy = np.copy(inputData)  # using numpy's copy method

    elif not inward and sign < 0:
        inputDataCopy = inputData * sign  # multiply all elements by sign
        icnt = inputDataSize

    elif inward and sign < 0:
        # Copy elements greater than x and multiply by sign
        for i in range(inputDataSize):
            if inputData[i] > x:
                inputDataCopy[icnt] = inputData[i] * sign
                icnt += 1

    elif inward and sign > 0:
        # Copy only elements less than x
        for i in range(inputDataSize):
            if inputData[i] < x:
                inputDataCopy[icnt] = inputData[i]
                icnt += 1

    inputDataCopy.sort()  # This sorts the array in ascending order in-place
    inputDataCopy = inputDataCopy[::-1]  # This reverses the array to descending order

    if scores_to_drop > 0:
        dataPtr = inputDataCopy[scores_to_drop:]  # Slicing to skip scores_to_drop elements
    else:
        dataPtr = inputDataCopy

    small_score = dataPtr[fitting_size - 1]  # Access the score at the fitting size index

    # Adjusting data values based on the translate_amount and subtracting the small_score
    dataPtr[:fitting_size] = dataPtr[:fitting_size] + translate_amount - small_score

    # Call to a hypothetical Weibull fitting function; substitute with actual function
    parmhat = np.zeros(2)  # Placeholder for Weibull parameters estimates
    parmci = np.zeros((2, 2))  # Placeholder for confidence intervals of the estimates
    rval = weibull_fit(parmhat, parmci, dataPtr[:fitting_size], alpha, fitting_size)

    # Set validity and handle fitting result
    is_valid = True
    if rval != 1:
        reset()  # Assuming reset() is defined elsewhere to reset some state

    return rval


class MetaRecognition:
    def __init__(self, scores_to_dropx, fitting_sizex, verb, alphax, translate_amountx):
        self.verbose = verb
        self.alpha = alphax
        self.fitting_size = fitting_sizex
        self.translate_amount = translate_amountx
        self.scores_to_drop = scores_to_dropx
        self.parmhat = np.zeros(2)  # Simulate the memset in C++
        self.parmci = np.zeros(4)  # Simulate the memset in C++
        self.sign = 1
        self.ftype = 'complement_reject'
        self.small_score = 0.0
        self.isvalid = False
        self.weibull_fit_verbose_debug = 1 if verb else 0

    def is_valid(self):
        return self.isvalid

    def set_translate(self, t):
        self.translate_amount = t
        self.isvalid = False

    def reset(self):
        self.parmhat.fill(0)  # np.zeros like initialization
        self.parmci.fill(0)  # np.zeros like initialization
        self.sign = 1
        self.scores_to_drop = 0
        self.small_score = 0.0
        self.isvalid = False

    def inv(self, x):
        if not self.isvalid:
            return -9999.0
        score = weibull_inv(x, self.parmhat[0], self.parmhat[1])  # Assume this function is defined
        return (score - self.translate_amount + self.small_score) * self.sign

    def cdf(self, x):
        if not self.isvalid:
            return -9999.0
        translated_x = x * self.sign + self.translate_amount - self.small_score
        wscore = weibull_cdf(translated_x, self.parmhat[0], self.parmhat[1])  # Assume this function is defined
        if self.ftype in ['complement_model', 'positive_model']:
            return 1 - wscore
        return wscore

    def pdf(self, x):
        if not self.isvalid:
            return -9999.0
        translated_x = x * self.sign + self.translate_amount - self.small_score
        prob = weibull_pdf(translated_x, self.parmhat[0], self.parmhat[1])  # Assume this function is defined
        if self.ftype in ['complement_model', 'positive_model']:
            return -prob
        return prob

    def w_score(self, x):
        return self.cdf(x)

    def predict_match(self, x, threshold):
        score = self.inv(threshold)
        if self.sign < 0:
            return x < score
        return x > score

    def re_normalize(self, invec, outvec, length):
        if not self.is_valid():
            return -9997

        rval = 1
        for i in range(length):
            outvec[i] = self.w_score(invec[i])

        return rval

    def evt_generic(self, inputData, inputDataSize, inward, x):
        if self.fitting_size > inputDataSize:
            print(
                f"In MetaRecognition, warning asked to fit with tail size {self.fitting_size} but input data only {inputDataSize}, returning -1",
                file=sys.stderr)
            return -1

        inputDataCopy = np.array(inputData)  # Copy the inputData to a numpy array for manipulation

        if not inward:
            if self.sign > 0:
                # Directly use inputDataCopy as is
                icnt = inputDataSize
            elif self.sign < 0:
                # Flip sign of all elements in inputDataCopy
                inputDataCopy *= self.sign
                icnt = inputDataSize
        else:
            if self.sign < 0:
                # Select elements greater than x and flip sign
                inputDataCopy = inputDataCopy[inputDataCopy > x] * self.sign
            elif self.sign > 0:
                # Select elements less than x
                inputDataCopy = inputDataCopy[inputDataCopy < x]
            icnt = len(inputDataCopy)

        # Sort data in descending order
        inputDataCopy = np.sort(inputDataCopy)[:icnt][::-1]  # Sorting and slicing for top scores

        if self.scores_to_drop > 0:
            dataPtr = inputDataCopy[self.scores_to_drop:]
        else:
            dataPtr = inputDataCopy

        # Ensure we don't access out of bounds if scores_to_drop + fitting_size > icnt
        if len(dataPtr) < self.fitting_size:
            print("Not enough data after dropping scores to fit the required size, returning -1", file=sys.stderr)
            return -1

        small_score = dataPtr[self.fitting_size - 1]

        # Translate and adjust scores
        dataPtr[:self.fitting_size] = dataPtr[:self.fitting_size] + self.translate_amount - small_score

        # Call fitting function and update validity
        rval = self.weibull_fit(self.parmhat, self.parmci, dataPtr[:self.fitting_size], self.alpha, self.fitting_size)
        self.isvalid = True
        if rval != 1:
            self.reset()

        return rval

def process_pdf_data(length):
    invec = np.random.normal(0, 1, length)  # Example input vector with normal distribution
    outvec = np.zeros(length)  # Pre-allocate output vector
    mr_instance = MetaRecognition()  # Assume an instance of MetaRecognition is created and configured somewhere

    if mr_instance.re_normalize_pdf(invec, outvec, length) != -9997:
        print("PDF normalization successful")
    else:
        print("PDF normalization failed, model is invalid")

    return outvec


FitHigh(12,1,3)


