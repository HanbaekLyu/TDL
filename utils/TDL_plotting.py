import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import matplotlib.font_manager as font_manager


###### Display functions

class TDL_plotting():
    def __init__(self,
                 W,
                 code,
                 A_full_predictions_trials,
                 df, # full data frame
                 input_variables,
                 full_state_list_train,
                 full_state_list_test,
                 data_test,  # test data array: states x days x variables
                 data_train, # train data array: states x days x variables
                 n_components=100,  # number of dictionary elements -- rank
                 patch_size=7,  # length of sliding window
                 prediction_length=1):

        self.W = W
        self.code = code
        self.A_full_predictions_trials = A_full_predictions_trials
        self.df = df
        self.input_variables = input_variables
        self.full_state_list_train = full_state_list_train
        self.full_state_list_test = full_state_list_test
        self.data_test = data_test
        self.data_train = data_train
        self.n_components = n_components
        self.patch_size = patch_size
        self.prediction_length = prediction_length


    def display_correlation(self, if_show, if_save, if_log_scale=False, filename=None,
                                    custom_code4ordering=None, pairwise=False, min=False, weighted_avg=False):

        W = self.W
        input_variables = self.input_variables
        k = self.patch_size
        x = self.data_test.shape
        rows = np.ceil(np.sqrt(W.shape[1])).astype(int)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
                                    subplot_kw={'xticks': [], 'yticks': []})

        code = self.code
        # print('code', code)
        importance = np.sum(code, axis=1) / sum(sum(code))

        if if_log_scale:
            W = np.exp(W) - 50

        if custom_code4ordering is None:
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
            idx = np.argsort(custom_importance)
            idx = np.flip(idx)

        dict_corr_array = np.zeros((W.shape[1], len(input_variables), len(input_variables)))
        for i in np.arange(W.shape[1]):
            dict = W[:, idx[i]].reshape(k, x[2])
            dict_df = pd.DataFrame(dict)

            dict_corr = dict_df.diff().corr()
            if dict_corr.isnull().values.any():
                dict_corr = np.ones((len(input_variables), len(input_variables)))
            dict_corr_array[i] = dict_corr


        if min:
            dict_corr_avg = np.min(dict_corr_array, axis=0)
        elif weighted_avg:
            dict_corr_array = np.array(dict_corr_array)
            custom_importance = np.array(importance)
            dict_corr_array_weighted = dict_corr_array * custom_importance.reshape((W.shape[1], 1, 1))
            dict_corr_avg = np.sum(dict_corr_array_weighted, axis=0)
        else:
            dict_corr_avg = np.mean(dict_corr_array, axis=0)

        if pairwise:
            return dict_corr_avg
        else:
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns_plot = sns.heatmap(dict_corr_avg, vmax=1, vmin=-1, square=True, cmap=cmap)
            figure = sns_plot.get_figure()

        if if_save:
            np.save(filename, dict_corr_avg)
            figure.savefig(filename + ".png")




    def display_dictionary(self, if_show, if_save, if_log_scale=False, filename=None,
                                    custom_code4ordering=None):

        for state_name in self.full_state_list_train:
            W = self.W
            input_variables = self.input_variables
            k = self.patch_size
            x = self.data_test.shape
            rows = np.ceil(np.sqrt(W.shape[1])).astype(int)

            fig, axs = plt.subplots(nrows=rows, ncols=rows, figsize=(10, 10),
                                    # subplot_kw={'xticks': [], 'yticks': []})
                                    subplot_kw={'xticks': []})

            print('W.shape', W.shape)

            code = self.code
            # print('code', code)
            importance = np.sum(code, axis=1) / sum(sum(code))

            if if_log_scale:
                W = np.exp(W) - 50

            if custom_code4ordering is None:
                idx = np.argsort(importance)
                idx = np.flip(idx)
            else:
                custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
                idx = np.argsort(custom_importance)
                idx = np.flip(idx)

            if (rows == 1):
                dict = W[:, idx[0]].reshape(x[0], k, x[2])
                j = self.full_state_list_train.index(state_name)
                marker_list = itertools.cycle(('*', 'x', '^', 'o', '|', '+'))


                for c in np.arange(dict.shape[2]):
                    variable_name = input_variables[c]
                    variable_name = variable_name.replace('input_', '')

                    axs.plot(np.arange(k), dict[j, :, c], marker=next(marker_list), label=variable_name)


                axs.set_xlabel('%1.2f' % importance[idx[0]], fontsize=13)  # get the largest first
                axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

            else:
                for axs, i in zip(axs.flat, range(W.shape[1])):
                    dict = W[:, idx[i]].reshape(x[0], k, x[2])
                    # print('x.shape', x)
                    j = self.full_state_list_train.index(state_name)
                    marker_list = itertools.cycle(('*', 'x', '^', 'o', '|', '+'))


                    for c in np.arange(dict.shape[2]):
                        variable_name = input_variables[c]
                        variable_name = variable_name.replace('input_', '')

                        axs.plot(np.arange(k), dict[j, :, c], marker=next(marker_list), label=variable_name)



                    axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
                    axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

            handles, labels = axs.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right')  ## bbox_to_anchor=(0,0)
            plt.suptitle(str(state_name) + '-Temporal Dictionary of size %d' % k, fontsize=16)
            # plt.subplots_adjust(left=0.01, right=0.55, bottom=0.05, top=0.99, wspace=0.1, hspace=0.4)  # for 24 atoms

            plt.subplots_adjust(left=0.01, right=0.62, bottom=0.1, top=0.8, wspace=0.1, hspace=0.4)  # for 12 atoms
            # plt.tight_layout()

            if if_save:
                if filename is None:
                    plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + str(state_name) + '.png')
                else:
                    plt.savefig(filename + '.png')
            if if_show:
                plt.show()

    def display_dictionary_old(self, cases, if_show, if_save, if_log_scale=False, filename=None, custom_code4ordering=None):
        W = self.W
        k = self.patch_size
        x = self.data_test.shape
        rows = np.floor(np.sqrt(W.shape[1])).astype(int)
        cols = np.ceil(np.sqrt(W.shape[1])).astype(int)

        '''
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(4, 3.5),
                                subplot_kw={'xticks': [], 'yticks': []})
        '''
        fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(4, 4.5),
                                subplot_kw={'xticks': [], 'yticks': []})

        print('W.shape', W.shape)

        code = self.code
        # print('code', code)
        importance = np.sum(code, axis=1) / sum(sum(code))

        if if_log_scale:
            W = np.exp(W) - 50

        if custom_code4ordering is None:
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
            idx = np.argsort(custom_importance)
            idx = np.flip(idx)

        # print('W', W)
        if cases == 'confirmed':
            c = 0
        elif cases == 'death':
            c = 1
        else:
            c = 2

        for axs, i in zip(axs.flat, range(W.shape[1])):
            dict = W[:, idx[i]].reshape(x[0], k, x[2])
            # print('x.shape', x)
            for j in np.arange(dict.shape[0]):
                country_name = self.country_list[j]
                marker = ''
                if country_name == 'Korea, South':
                    marker = '*'
                elif country_name == 'China':
                    marker = 'x'
                elif country_name == 'US':
                    marker = '^'
                axs.plot(np.arange(k), dict[j, :, c], marker=marker, label='' + str(country_name))
            axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
            axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')  ## bbox_to_anchor=(0,0)
        # plt.suptitle(cases + '-Temporal Dictionary of size %d'% k, fontsize=16)
        # plt.subplots_adjust(left=0.01, right=0.55, bottom=0.05, top=0.99, wspace=0.1, hspace=0.4)  # for 24 atoms

        plt.subplots_adjust(left=0.01, right=0.62, bottom=0.1, top=0.99, wspace=0.1, hspace=0.4)  # for 12 atoms
        # plt.tight_layout()

        if if_save:
            if filename is None:
                plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + cases + '.png')
            else:
                plt.savefig(
                    'Time_series_dictionary/' + str(foldername) + '/Dict-' + cases + '_' + str(filename) + '.png')
        if if_show:
            plt.show()

    def display_dictionary_single(self, if_show, if_save, foldername, filename, if_log_scale=False, custom_code4ordering=None):
        W = self.W
        k = self.patch_size
        x = self.data_test.shape
        code = self.code
        # print('code', code)
        importance = np.sum(code, axis=1) / sum(sum(code))

        if if_log_scale:
            W = np.exp(W) - 50

        if custom_code4ordering is None:
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
            idx = np.argsort(custom_importance)
            idx = np.flip(idx)

        # rows = np.floor(np.sqrt(W.shape[1])).astype(int)
        # cols = np.ceil(np.sqrt(W.shape[1])).astype(int)
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(4, 3),
                                subplot_kw={'xticks': [], 'yticks': []})
        print('W.shape', W.shape)
        # print('W', W)
        for axs, i in zip(axs.flat, range(W.shape[1])):
            for c in np.arange(x[2]):
                if c == 0:
                    cases = 'confirmed'
                elif c == 1:
                    cases = 'death'
                else:
                    cases = 'recovered'

                dict = W[:, idx[i]].reshape(x[0], k, x[2])  ### atoms with highest importance appears first
                for j in np.arange(dict.shape[0]):

                    if c == 0:
                        marker = '*'
                    elif c == 1:
                        marker = 'x'
                    else:
                        marker = 's'

                    axs.plot(np.arange(k), dict[j, :, c], marker=marker, label='' + str(cases))
                axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=14)  # get the largest first
                axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')  ## bbox_to_anchor=(0,0)
        # plt.suptitle(str(self.country_list[0]) + '-Temporal Dictionary of size %d'% k, fontsize=16)
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.3, top=0.99, wspace=0.1, hspace=0.4)
        # plt.tight_layout()

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + str(self.country_list[0]) + '_' + str(
                filename) + '.png')
        if if_show:
            plt.show()

    def display_prediction_evaluation(self, filename, if_show=False, if_save=True, if_log_scale=False, if_errorbar=True,
                                      if_evaluation=True, title=None):

        prediction = self.A_full_predictions_trials
        input_variables = self.input_variables
        A = self.data_test
        k = self.patch_size
        A_recons = prediction
        print('!!!!!!!A_recons.shape', A_recons.shape)
        A_predict = A_recons.copy()

        if if_evaluation:
            A_predict1 = A_recons.copy()
            ### A_recons.shape = (# trials) x (# days) x (# states) x (Future Extrapolation Length) x (# variables)
            A_recons1 = np.zeros(shape=(A_predict1.shape[0], A.shape[1] + A_predict1.shape[3], A.shape[0], A.shape[2]))
            A_recons1 = A_predict1[:, :, :, -1, :]

            # for i in np.arange(0, A_predict1.shape[2]):
            #     A_recons1[:,i + A_predict1.shape[3],:,:] = A_predict1[:,i,:, -1,:]
            # We are making d-days ahead prediction where d = (Future Extrapolation Length) + (prediction_length) -1
            a = np.zeros(shape=(A_predict1.shape[0], A_predict1.shape[3] + self.prediction_length-1, A.shape[0], A.shape[2]))
            A_recons1 = np.append(a, A_recons1, axis=1)  # Shift A_recons1 by d in time
            # print('!!!! A.shape[1]+A_predict1.shape[3]', A.shape[1] + A_predict1.shape[3] + self.prediction_length)
            # print('!!!! A_recons1.shape', A_recons1.shape)

            A_recons1 = np.swapaxes(A_recons1, axis1=1, axis2=2)
            # fill in first d entries of prediction by the raw data
            for trial in np.arange(0, A_predict1.shape[0]):
                for j in np.arange(0, A_predict1.shape[3] + self.prediction_length):
                    A_recons1[trial, :, j, :] = A[:, j, :]

            A_recons = A_recons1
            A_predict = A_recons.copy()
            print('!!!!!!!! A_recons.shape', A_recons.shape)

        if if_log_scale:
            A = np.exp(A) - 50
            A_recons = np.exp(A_recons) - 50

        if if_errorbar:
            # print('!!!', A_predict.shape)  # trials x states x days x variables
            A_predict = np.sum(A_recons, axis=0) / A_recons.shape[0]  ### axis-0 : trials
            A_std = np.std(A_recons, axis=0)
            # print('!!! A_std', A_std)

        ### Make gridspec
        fig1 = plt.figure(figsize=(15, 10), constrained_layout=False)
        gs1 = fig1.add_gridspec(nrows=A_predict.shape[2], ncols=A_predict.shape[0], wspace=0.2, hspace=0.2)

        # font = font_manager.FontProperties(family="Times New Roman", size=11)

        for i in range(A_predict.shape[0]):
            for c in range(A_predict.shape[2]):

                ax = fig1.add_subplot(gs1[c, i])

                variable_name = input_variables[c]
                variable_name = variable_name.replace('input_', '')

                ### get days xticks
                try:
                    start_day = self.df.get(self.full_state_list_train[0]).index[0]
                except:
                    start_day = self.df.index[0]
                x_data = pd.date_range(start_day, periods=A.shape[1], freq='D')
                x_data_recons = pd.date_range(start_day, periods=A_predict.shape[1] - self.patch_size, freq='D')
                x_data_recons += pd.DateOffset(self.patch_size)

                ### plot axs
                ax.plot(x_data, A[i, :, c], 'b-', marker='o', markevery=5, label='Original-' + str(variable_name))

                if not if_errorbar:
                    ax.plot(x_data_recons, A_predict[i, self.patch_size:A_predict.shape[1], c],
                            'r-', marker='x', markevery=5, label='Prediction-' + str(variable_name))
                else:
                    markers, caps, bars = ax.errorbar(x_data_recons,
                                                      A_predict[i, self.patch_size:A_predict.shape[1], c],
                                                      yerr=A_std[i, self.patch_size:A_predict.shape[1], c],
                                                      fmt='r-', label='Prediction-' + str(variable_name), errorevery=1)

                    [bar.set_alpha(0.5) for bar in bars]
                    # [cap.set_alpha(0.5) for cap in caps]

                ax.set_ylim(0, np.maximum(np.max(A[i, :, c]), np.max(A_predict[i, :, c] + A_std[i, :, c])) * 1.1)

                if c == 0:
                    if title is None:
                        ax.set_title(str(self.state_list[i]), fontsize=15)
                    else:
                        ax.set_title(title, fontsize=15)

                ax.yaxis.set_label_position("left")
                # ax.yaxis.set_label_coords(0, 2)
                # ax.set_ylabel(str(list[j]), rotation=90)
                ax.set_ylabel('population', fontsize=10)  # get the largest first
                ax.yaxis.set_label_position("left")
                ax.legend()

        fig1.autofmt_xdate()
        # fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19 : '+ str(self.country_list[0]) +
        #             "\n seg. length = %i, # temp. dict. atoms = %i, learning exponent = %1.3f" % (self.patch_size, W.shape[1], self.beta),
        #             fontsize=12, y=0.96)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig(filename + '.pdf')
        if if_show:
            plt.show()

    def display_prediction_single(self, source, prediction, if_show, if_save, foldername, filename):
        A = self.combine_data(source)[0]
        k = self.patch_size
        A_predict = prediction

        if self.if_log_scale:
            A = np.exp(A) - 50
            A_predict = np.exp(A_predict) - 50

        fig, axs = plt.subplots(nrows=A.shape[2], ncols=1, figsize=(5, 5))
        lims = [(np.datetime64('2020-01-21'), np.datetime64('2020-07-15')),
                (np.datetime64('2020-01-21'), np.datetime64('2020-07-15')),
                (np.datetime64('2020-01-21'), np.datetime64('2020-07-15'))]
        if A.shape[2] == 1:
            L = zip([axs], np.arange(A.shape[2]))
        else:
            L = zip(axs.flat, np.arange(A.shape[2]))
        for axs, c in L:
            if c == 0:
                cases = 'confirmed'
            elif c == 1:
                cases = 'death'
            else:
                cases = 'recovered'

            ### get days xticks
            x_data = pd.date_range('2020-01-21', periods=A.shape[1], freq='D')
            x_data_recons = pd.date_range('2020-01-21', periods=A_predict.shape[1] - self.patch_size, freq='D')
            x_data_recons += pd.DateOffset(self.patch_size)

            ### plot axs
            axs.plot(x_data, A[0, :, c], 'b-', marker='o', markevery=5, label='Original-' + str(cases))
            axs.plot(x_data_recons, A_predict[0, self.patch_size:A_predict.shape[1], c],
                     'r-', marker='x', markevery=5, label='Prediction-' + str(cases))
            axs.set_ylim(0, np.max(A_predict[0, :, c]) + 10)

            # ax.text(2, 0.65, str(list[j]))
            axs.yaxis.set_label_position("right")
            # ax.yaxis.set_label_coords(0, 2)
            # ax.set_ylabel(str(list[j]), rotation=90)
            axs.set_ylabel('log(population)', fontsize=10)  # get the largest first
            axs.yaxis.set_label_position("right")
            axs.legend(fontsize=11)

        fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19 : ' + str(self.country_list[0]) +
                     "\n seg. length = %i, # temp. dict. atoms = %i, learning exponent = %1.3f" % (
                         self.patch_size, W.shape[1], self.beta),
                     fontsize=12, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Plot-' + str(self.country_list[0]) + '-' + str(
                filename) + '.png')
        if if_show:
            plt.show()

    def display_prediction(self, source, prediction, cases, if_show, if_save, foldername, if_errorbar=False):
        A = self.combine_data(source)[0]
        k = self.patch_size
        A_recons = prediction
        A_predict = prediction

        if self.if_log_scale:
            A = np.exp(A) - 50
            A_recons = np.exp(A_recons) - 50
            A_predict = np.exp(A_predict) - 50

        A_std = np.zeros(shape=A_recons.shape)
        if if_errorbar:
            A_predict = np.sum(A_predict, axis=0) / A_predict.shape[0]  ### axis-0 : trials
            A_std = np.std(A_recons, axis=0)
            # print('A_std', A_std)

        L = len(self.country_list)  # number of countries
        rows = np.floor(np.sqrt(L)).astype(int)
        cols = np.ceil(np.sqrt(L)).astype(int)

        ### get days xticks
        x_data = pd.date_range('2020-01-21', periods=A.shape[1], freq='D')
        x_data_recons = pd.date_range('2020-01-21', periods=A_predict.shape[1] - self.patch_size, freq='D')
        x_data_recons += pd.DateOffset(self.patch_size)

        if cases == 'confirmed':
            c = 0
        elif cases == 'death':
            c = 1
        else:
            c = 2

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 5))
        for axs, j in zip(axs.flat, range(L)):
            country_name = self.country_list[j]
            if self.country_list[j] == 'Korea, South':
                country_name = 'Korea, S.'

            axs_empty = axs.plot([], [], ' ', label=str(country_name))
            axs_original = axs.plot(x_data, A[j, :, c], 'b-', marker='o', markevery=5, label='Original')
            if not if_errorbar:
                axs_recons = axs.plot(x_data_recons, A_predict[j, self.patch_size:A_predict.shape[1], c],
                                      'r-', marker='x', markevery=5, label='Prediction-' + str(country_name))
            else:
                y = A_predict[j, self.patch_size:A_predict.shape[1], c]
                axs_recons = axs.errorbar(x_data_recons, y, yerr=A_std[j, self.patch_size:A_predict.shape[1], c],
                                          fmt='r-.', label='Prediction', errorevery=2, )
            axs.set_ylim(0, np.maximum(np.max(A[j, :, c]), np.max(A_predict[j, :, c] + A_std[j, :, c])) * 1.1)

            # ax.text(2, 0.65, str(list[j]))
            axs.yaxis.set_label_position("right")
            # ax.yaxis.set_label_coords(0, 2)
            # ax.set_ylabel(str(list[j]), rotation=90)

            axs.legend(fontsize=9)

            fig.autofmt_xdate()
            fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19:' + cases +
                         "\n segment length = %i, # temporal dictionary atoms = %i" % (
                             self.patch_size, W.shape[1]),
                         fontsize=12, y=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Plot-' + cases + '.png')
        if if_show:
            plt.show()
