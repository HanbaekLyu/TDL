from utils.onmf import Online_NMF
import numpy as np
from sklearn.decomposition import SparseCoder
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import matplotlib.font_manager as font_manager
# import covid_dataprocess
import itertools

DEBUG = False

# Temporal Dictionary Learning for general time-series data (not specific to COVID19 data)
# Learns "elementary time-series patterns" by NMF from slices of time-series data
# Aurthor : Hanbaek Lyu
# Reference: Hanbaek Lyu, Christopher Strohmeier, Deanna Needell, and Georg Menz,
# “COVID-19 Time Series Prediction by Joint Dictionary Learning and Online NMF”
# https://arxiv.org/abs/2004.09112
# See also:  https://github.com/HanbaekLyu/ONMF-COVID19 for COVID19-sepecific implementation

class TDL():
    def __init__(self,
                 data_test,  # test data array: states x days x variables
                 data_train, # train data array: states x days x variables
                 n_components=100,  # number of dictionary elements -- rank
                 ini_dict=None,
                 ONMF_iterations=50,  # number of iterations for the ONMF algorithm
                 ONMF_sub_iterations=20,  # number of i.i.d. subsampling for each iteration of onmf
                 ONMF_batch_size=20,  # number of patches used in i.i.d. subsampling
                 num_patches_perbatch=1000,  # number of patches that onmf algorithm learns from at each iteration
                 patch_size=7,  # length of sliding window
                 patches_file='',
                 learn_joint_dict=False,
                 prediction_length=1,
                 learnevery=5,
                 learning_window_cap=None,  # if not none, learn from the past "learning_window_cap" days to predict
                 alpha=None,
                 beta=None,
                 subsample=False):
        '''
        batch_size = number of patches used for training dictionaries per onmf iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''

        self.data_test = data_test
        self.data_train = data_train
        self.n_components = n_components
        self.ONMF_iterations = ONMF_iterations
        self.ONMF_sub_iterations = ONMF_sub_iterations
        self.num_patches_perbatch = num_patches_perbatch
        self.ONMF_batch_size = ONMF_batch_size
        self.patch_size = patch_size
        self.patches_file = patches_file
        self.learn_joint_dict = learn_joint_dict
        self.prediction_length = prediction_length
        self.code = np.zeros(shape=(n_components, num_patches_perbatch))
        self.learnevery = learnevery
        self.alpha = alpha
        self.beta = beta
        self.subsample = subsample
        self.result_dict = {}
        self.learning_window_cap = learning_window_cap

        input_variable_list = []

        # print('data', self.data_test)
        # print('!!! data.shape', self.data_test.shape)
        self.nmf = Online_NMF(self.data_test, self.n_components,
                              iterations=self.ONMF_sub_iterations,
                              learn_joint_dict=True,
                              mode=3,
                              ini_dict=None,
                              ini_A=None,
                              ini_B=None,
                              batch_size=self.ONMF_batch_size)
        self.W = ini_dict
        if ini_dict is None:
            self.W = np.ones(shape=(self.data_test.shape[0] * self.data_test.shape[2] * patch_size, n_components))


    def extract_random_patches(self, batch_size=None, time_interval_initial=None, A=None):
        '''
        Extract 'num_patches_perbatch' (segments) of size 'patch_size'many random patches of given size
        '''
        x = self.data_test.shape  # shape = 2 (ask, bid) * time * country
        k = self.patch_size
        if batch_size is None:
            num_patches_perbatch = self.num_patches_perbatch
        else:
            num_patches_perbatch = batch_size

        X = np.zeros(shape=(x[0], k, x[2], 1))  # 1 * window length * country * num_patches_perbatch
        for i in np.arange(num_patches_perbatch):
            if time_interval_initial is None:
                a = np.random.choice(x[1] - k)  # starting time of a window patch of length k
            else:
                a = time_interval_initial + i
            if A is None:
                Y = self.data_train[:, a:a + k, :]  # shape 2 * k * x[2]
            else:
                Y = A[:, a:a + k, :]  # shape 2 * k * x[2]

            Y = Y[:, :, :, np.newaxis]
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X  # X.shape = (2, k, num_countries, num_patches_perbatch)

    def extract_patches_interval(self, time_interval_initial, time_interval_terminal, A=None):
        '''
        Extract a given number of patches (segments) of size 'patch_size' during the given interval
        X.shape = (# states) x (# window length) x (# variables) x (num_patches_perbatch)
        '''
        x = self.data_test.shape  # shape = (# states) x (# days) x (# variables)
        k = self.patch_size  # num of consecutive days to form a single patch = window length

        X = np.zeros(
            shape=(x[0], k, x[2], 1))  # (# states) x (# window length) x (# variables) x (num_patches_perbatch)

        # print('>>>>> [time_interval_initial, time_interval_terminal]', [time_interval_initial, time_interval_terminal])

        for i in np.arange(self.num_patches_perbatch):
            a = np.random.choice(np.arange(time_interval_initial, time_interval_terminal - k + 1))
            if A is None:
                Y = self.data_train[:, a:a + k, :]  # shape: # idx * k * x[2]
            else:
                Y = A[:, a:a + k, :]  # shape: # idx * k * x[2]


            Y = Y[:, :, :, np.newaxis]
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X

    def data_to_patches(self):
        '''

        args:
            path (string): Path and filename of input time series data
            patch_size (int): length of sliding window we are extracting from the time series (data)
        returns:

        '''

        if DEBUG:
            print(np.asarray(self.data_test))

        patches = self.extract_random_patches()
        print('patches.shape=', patches.shape)
        return patches

    def train_dict(self,
                   foldername,
                   mode=1,
                   alpha=0,
                   beta=1,
                   learn_joint_dict=True,
                   data_train = None,
                   iterations=None,
                   update_self=True,
                   nonnegative_code=True,
                   nonnegative_dict=True,
                   dict_sparsity=0,
                   code_sparsity=0,
                   if_save=False,
                   print_iter=False):
        # print('training dictionaries from patches along mode %i...' % mode)
        '''
        Trains dictionary based on patches from an i.i.d. sequence of batch of patches
        mode = 0, 1, 2
        learn_joint_dict = True or False parameter
        '''
        W = self.W
        At = []
        Bt = []
        code = self.code
        if data_train is None:
            data_train = self.data_train

        if iterations is not None:
            n_iter = iterations
        else:
            n_iter = self.ONMF_iterations

        for_list = np.arange(n_iter)
        if print_iter:
            for_list = trange(n_iter)

        for t in for_list:
            X = self.extract_random_patches(A=data_train)  ## need to sample patches from self.data_test
            if t == 0:
                self.nmf = Online_NMF(X, self.n_components,
                                      ini_dict=W,
                                      iterations=self.ONMF_sub_iterations,
                                      learn_joint_dict=learn_joint_dict,
                                      mode=mode,
                                      beta=beta,
                                      batch_size=self.ONMF_batch_size)  # max number of possible patches
                W, At, Bt, H = self.nmf.train_dict_single(nonnegative_code=nonnegative_code,
                                                          nonnegative_dict=nonnegative_dict,
                                                          dict_sparsity=dict_sparsity,
                                                          code_sparsity=code_sparsity)
                code += H
            else:
                self.nmf = Online_NMF(X, self.n_components,
                                      iterations=self.ONMF_sub_iterations,
                                      batch_size=self.ONMF_batch_size,
                                      ini_dict=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      beta=beta,
                                      learn_joint_dict=learn_joint_dict,
                                      mode=mode,
                                      history=self.nmf.history)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.nmf.train_dict_single(nonnegative_code=nonnegative_code,
                                                          nonnegative_dict=nonnegative_dict,
                                                          dict_sparsity=dict_sparsity,
                                                          code_sparsity=code_sparsity)
                code += H

            #if print_iter:
            #    print('Current minibatch training iteration %i out of %i' % (t, self.ONMF_iterations))

        if update_self:
            self.W = W
            self.code = code

        # print('dict_shape:', W.shape)
        # print('code_shape:', code.shape)
        if if_save:
            np.save('Time_series_dictionary/' + str(foldername) + '/dict_learned_' + str(
                mode) + '_' + 'pretraining' , self.W)
            np.save('Time_series_dictionary/' + str(foldername) + '/code_learned_' + str(
                mode) + '_' + 'pretraining' , self.code)
            # np.save('Time_series_dictionary/' + str(foldername) + '/At_' + str(mode) + '_' + 'pretraining' + '_' + str(list[0]), At)
            # np.save('Time_series_dictionary/' + str(foldername) + '/Bt_' + str(mode) + '_' + 'pretraining' + '_' + str(list[0]), Bt)
        return W, At, Bt, code

    def ONMF_predictor(self,
                       mode,
                       foldername,
                       data_test=None,
                       data_train=None,
                       prelearned_dict = None, # if not none, use this dictionary for prediction
                       ini_dict=None,
                       ini_A=None,
                       ini_B=None,
                       beta=1,
                       a1=0,  # regularizer for the code in partial fitting
                       a2=0,  # regularizer for the code in recursive prediction
                       future_extrapolation_length=0,
                       if_learn_online=True,
                       if_save=True,
                       # if_recons=True,  # Reconstruct observed data using learned dictionary
                       learning_window_cap = None, # if not none, learn only from the past "learning_window_cap" days
                       minibatch_training_initialization=True,
                       minibatch_alpha=1,
                       minibatch_beta=1,
                       print_iter=False,
                       online_learning=True,
                       num_trials=1):
        # print('online learning and predicting from patches along mode %i...' % mode)
        '''
        Trains dictionary along a continuously sliding window over the data stream
        Predict forthcoming data on the fly. This could be made to affect learning rate
        '''

        if data_test is None:
            data_test = self.data_test.copy()

        if data_train is None:
            data_train = self.data_train.copy()



        if learning_window_cap is None:
            learning_window_cap = self.learning_window_cap

        # print('!!!!!!!!!! A.shape', A.shape)

        k = self.patch_size  # Window length
        L = self.prediction_length
        # A_recons = np.zeros(shape=A.shape)
        # print('A_recons.shape',A_recons.shape)
        # W = self.W
        # print('W.shape', self.W.shape)
        At = []
        Bt = []
        H = []
        # A_recons = np.zeros(shape=(A.shape[0], k+L-1, A.shape[2]))

        list_full_predictions = []
        A_recons = data_test.copy()

        for trial in np.arange(num_trials):
            ### Initialize self parameters
            self.W = ini_dict
            # A_recons = A[:, 0:k + L - 1, :]
            At = []
            Bt = []

            if prelearned_dict is not None:
                self.W = prelearned_dict
            else:
                # Learn new dictionary to use for prediction
                if minibatch_training_initialization:
                    # print('!!! self.W right before minibatch training', self.W)
                    self.W, At, Bt, H = self.train_dict(mode=3,
                                                        alpha=minibatch_alpha,
                                                        beta=minibatch_beta,
                                                        iterations=self.ONMF_iterations,
                                                        learn_joint_dict=True,
                                                        foldername=None,
                                                        update_self=True,
                                                        if_save=False)

                # print('data.shape', self.data_test.shape)
                # iter = np.floor(A.shape[1]/self.num_patches_perbatch).astype(int)
                if online_learning:
                    T_start = k
                    if learning_window_cap is not None:
                        T_start = max(0, data_train.shape[1]-k-learning_window_cap)
                    else:
                        T_start = 0

                    interval = range(T_start, data_train.shape[1]-k)

                    initial_iter = True
                    for t in interval:
                        X = self.extract_patches_interval(time_interval_initial=t,
                                                          time_interval_terminal=t + k,
                                                          A = data_train)

                        # X.shape = (# states) x (# window length) x (# variables) x (num_patches_perbatch)

                        if initial_iter:
                            self.nmf = Online_NMF(X, self.n_components,
                                                  iterations=self.ONMF_sub_iterations,
                                                  learn_joint_dict=True,
                                                  mode=mode,
                                                  ini_dict=self.W,
                                                  ini_A=ini_A,
                                                  ini_B=ini_B,
                                                  batch_size=self.ONMF_batch_size,
                                                  subsample=self.subsample,
                                                  beta=beta)
                            self.W, At, Bt, H = self.nmf.train_dict_single()
                            self.code += H

                            initial_iter = False

                        else:
                            if t % self.learnevery == 0 and if_learn_online:  # do not learn from zero data (make np.sum(X)>0 for online learning)
                                self.nmf = Online_NMF(X, self.n_components,
                                                      iterations=self.ONMF_sub_iterations,
                                                      batch_size=self.ONMF_batch_size,
                                                      ini_dict=self.W,
                                                      ini_A=At,
                                                      ini_B=Bt,
                                                      learn_joint_dict=True,
                                                      mode=mode,
                                                      history=self.nmf.history,
                                                      subsample=self.subsample,
                                                      beta=beta)

                                self.W, At, Bt, H = self.nmf.train_dict_single()
                                # print('dictionary_updated')
                                self.code += H


                        if print_iter:
                            print('Current (trial, day) for ONMF_predictor (%i, %i) out of (%i, %i)' % (
                                trial + 1, t, num_trials, data_train.shape[1] - 1))

                        # print('!!!!! A_recons.shape', A_recons.shape)

                # concatenate state-wise dictionary to predict one state
                # Assumes len(list_states)=1
                # print('!!!!!! self.W.shape', self.W.shape)


            #### forward recursive prediction begins
            for t in np.arange(data_test.shape[1], data_test.shape[1] + future_extrapolation_length):
                patch = A_recons[:, t - k + L:t, :]

                if t == data_test.shape[1]:
                    patch = data_test[:, t - k + L:t, :]

                # print('!!!!! patch.shape', patch.shape)
                '''
                error occurs
                '''
                try:
                    patch_recons = self.predict_joint_single(patch, a2)
                except:
                    print(t)
                    continue
                # print('!!!!! patch_recons.shape', patch_recons.shape)
                A_recons = np.append(A_recons, patch_recons, axis=1)
            # print('new cases predicted final', A_recons[0, -1, 0])

            ### initial regulation
            A_recons[:, 0:self.learnevery + L, :] = data_test[:, 0:self.learnevery + L, :]
            ### patch the two reconstructions
            # A_recons = np.append(A_recons, A_recons[:,A.shape[1]:, :], axis=1)

            # print('!!!!! A_recons.shape', A_recons.shape)

            list_full_predictions.append(A_recons.copy())

        A_full_predictions_trials = np.asarray(
            list_full_predictions)  ## shape = (# trials) x (# states) x (# days + L) x (# varibles)

        self.result_dict.update({'Evaluation_num_trials': str(num_trials)})
        self.result_dict.update({'Evaluation_A_full_predictions_trials': A_full_predictions_trials})
        self.result_dict.update({'Evaluation_Dictionary': self.W})
        self.result_dict.update({'Evaluation_Code': self.code})

        if if_save:
            np.save('Time_series_dictionary/' + str(foldername) + '/full_results_' + 'num_trials_' + str(num_trials), self.result_dict)

        return A_full_predictions_trials, self.W, At, Bt, self.code

    def ONMF_predictor_historic(self,
                                mode,
                                foldername,
                                prelearned_dict_seq = None, # if not none, use this seq of dict for prediction
                                ini_dict=None,
                                ini_A=None,
                                ini_B=None,
                                beta=1,
                                a1=0,  # regularizer for the code in partial fitting
                                a2=0,  # regularizer for the code in recursive prediction
                                future_extrapolation_length=0,
                                learning_window_cap = None,
                                if_save=True,
                                minibatch_training_initialization=False,
                                minibatch_alpha=1,
                                minibatch_beta=1,
                                online_learning=True,
                                num_trials=1):  # take a number of trials to generate empirical confidence interval

        print('Running ONMF_timeseries_predictor_historic along mode %i...' % mode)
        '''
        Apply online_learning_and_prediction for intervals [0,t] for every 1\le t\le T to make proper all-time predictions
        for evaluation
        '''

        A = self.data_test

        # print('A.shape', A.shape)
        k = self.patch_size
        L = self.prediction_length
        FEL = future_extrapolation_length
        # A_recons = np.zeros(shape=A.shape)
        # print('A_recons.shape',A_recons.shape)
        # W = self.W
        if learning_window_cap is None:
            learning_window_cap = self.learning_window_cap

        self.W = ini_dict
        if ini_dict is None:
            d = self.data_test.shape[0]*k*self.data_test.shape[2]     #(# states) x (# window length) x (# variables)
            self.W = np.random.rand(d, self.n_components)
        # print('W.shape', self.W.shape)

        # A_recons = np.zeros(shape=(A.shape[0], k+L-1, A.shape[2]))
        # A_recons = A[:, 0:k + L - 1, :]

        list_full_predictions = []
        W_total_seq_trials = []
        for trial in trange(num_trials):
            W_total_seq = []
            ### A_total_prediction.shape = (# days) x (# states) x (FEL) x (# variables)
            ### W_total_seq.shape = (# days) x (# states * window length * # variables) x (n_components)
            A_total_prediction = []
            ### fill in predictions for the first k days with the raw data
            for i in np.arange(k + 1):
                A_total_prediction.append(A[:, i:i + FEL, :])
                W_total_seq.append(self.W.copy())
            for t in trange(k + 1, A.shape[1]):
                ### Set self.data_test to the truncated one during [1,t]
                prelearned_dict = None
                if prelearned_dict_seq is not None:
                    prelearned_dict = prelearned_dict_seq[trial,t,:,:]

                A_recons, W, At, Bt, code = self.ONMF_predictor(mode,
                                                                foldername,
                                                                data_test=self.data_test[:, :t, :],
                                                                data_train=self.data_train[:, :t, :],
                                                                prelearned_dict=prelearned_dict,
                                                                ini_dict=ini_dict,
                                                                ini_A=ini_A,
                                                                ini_B=ini_B,
                                                                beta=beta,
                                                                a1=a1,
                                                                # regularizer for the code in partial fitting
                                                                a2=a2,
                                                                # regularizer for the code in recursive prediction
                                                                future_extrapolation_length=future_extrapolation_length,
                                                                learning_window_cap=learning_window_cap,
                                                                if_save=True,
                                                                minibatch_training_initialization=minibatch_training_initialization,
                                                                minibatch_alpha=minibatch_alpha,
                                                                minibatch_beta=minibatch_beta,
                                                                print_iter=False,
                                                                online_learning=online_learning,
                                                                num_trials=1)

                A_recons = A_recons[0, :, :, :]
                # print('!!!! A_recons.shape', A_recons.shape)
                ### A_recons.shape = (# states, t+FEL, # variables)
                # print('!!!!! A_recons[:, -FEL:, :].shape', A_recons[:, -FEL:, :].shape)
                A_total_prediction.append(A_recons[:, -FEL:, :])
                W_total_seq.append(W.copy())
                ### A_recons.shape = (# states, t+FEL, # variables)
                # print('Current (trial, day) for ONMF_predictor_historic (%i, %i) out of (%i, %i)' % (trial + 1, t - k, num_trials, A.shape[1] - k - 1))

            A_total_prediction = np.asarray(A_total_prediction)
            W_total_seq = np.asarray(W_total_seq)
            print('W_total_seq.shape', W_total_seq.shape)
            W_total_seq_trials.append(W_total_seq)
            list_full_predictions.append(A_total_prediction)

        W_total_seq_trials = np.asarray(W_total_seq_trials)
        A_full_predictions_trials = np.asarray(list_full_predictions)
        print('!!! A_full_predictions_trials.shape', A_full_predictions_trials.shape)

        self.result_dict.update({'Evaluation_num_trials': str(num_trials)})
        self.result_dict.update({'Evaluation_A_full_predictions_trials': A_full_predictions_trials})
        self.result_dict.update({'Evaluation_Dictionary_seq_trials': W_total_seq_trials})
        # sequence of dictionaries to be used for historic prediction : shape (trials, time, W.shape[0], W.shape[1])

        if if_save:

            np.save('Time_series_dictionary/' + str(foldername) + '/full_results_' + 'num_trials_' + str(
                num_trials), self.result_dict)

    def predict_joint_single(self, data, a1):
        # print('!!!! self.W.shape', self.W.shape)
        # self.W = np.concatenate(np.vsplit(self.W, self.data_test.shape[0]), axis=1)
        k = self.patch_size
        L = self.prediction_length
        A = data  # A.shape = (self.data_test.shape[0], k-L, self.data_test.shape[2])
        # A_recons = np.zeros(shape=(A.shape[0], k, A.shape[2]))
        # W_tensor = self.W.reshape((k, A.shape[0], -1))
        # print('A.shape', A.shape)
        W_tensor = self.W.reshape((self.data_test.shape[0], k, self.data_test.shape[2], -1))
        # print('W.shape', W_tensor.shape)

        # for missing data, not needed for the COVID-19 data set
        # extract only rows of nonnegative values (disregarding missing entries) (negative = N/A)

        J = np.where(np.min(A, axis=(0, 1)) >= 0)
        A_pos = A[:, :, J]
        # print('A_pos', A_pos)
        # print('np.min(A)', np.min(A))
        W_tensor = W_tensor[:, :, J, :]
        W_trimmed = W_tensor[:, 0:k - L, :, :]
        W_trimmed = W_trimmed.reshape((-1, self.n_components))
        # print('!!! W_trimmed.shape', W_trimmed.shape)

        patch = A_pos

        # print('patch', patch)

        patch = patch.reshape((-1, 1))
        # print('!!!!! patch.shape', patch.shape)

        # print('patch', patch)

        coder = SparseCoder(dictionary=W_trimmed.T, transform_n_nonzero_coefs=None,
                            transform_alpha=a1, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter
        code = coder.transform(patch.T)
        patch_recons = np.dot(self.W, code.T).T  # This gives prediction on the last L missing entries
        patch_recons = patch_recons.reshape(-1, k, A.shape[2])

        # now paint the reconstruction canvas
        # only add the last predicted value
        A_recons = patch_recons[:, -1, :]
        return A_recons[:, np.newaxis, :]


    def coding_historic(self,
                        ini_dict=None,
                        a1=0,
                        nonnagative_code=True):  # regularizer for the code in partial fitting


        print('Coding time series using joing temporal dictionary...')
        '''
        Code every temporal patch of length self.patch_size for every 1\le t\le T using the learned TD
        '''

        A = self.data_train
        k = self.patch_size

        W = ini_dict
        if ini_dict is None:
            W = self.W

        list_codes = []
        ### A_total_prediction.shape = (# days) x (# states) x (FEL) x (# variables)
        ### W_total_seq.shape = (# days) x (# states * window length * # variables) x (n_components)

        for t in trange(k, A.shape[1]):
            ### Set self.data_test to the truncated one during [1,t]

            X = A[:,t-k:t,:].reshape(-1,1)

            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=a1, transform_algorithm='lasso_lars', positive_code=nonnagative_code)
            # alpha = L1 regularization parameter.
            H = coder.transform(X.T).T
            list_codes.append(H)
        return list_codes
