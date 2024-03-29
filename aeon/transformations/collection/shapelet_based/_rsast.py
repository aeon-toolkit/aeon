import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.numba.general import z_normalise_series
from aeon.utils.validation import check_n_jobs



@njit(fastmath=False)
def apply_kernel(ts, arr):
    d_best = np.inf  # sdist
    m = ts.shape[0]
    kernel = arr[~np.isnan(arr)]  # ignore nan

    # profile = mass2(ts, kernel)
    # d_best = np.min(profile)

    l = kernel.shape[0]
    for i in range(m - l + 1):
        d = np.sum((z_normalise_series(ts[i:i+l]) - kernel)**2)
        if d < d_best:
            d_best = d

    return d_best


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i] = apply_kernel(ts, k)
    return out


class RSAST(BaseCollectionTransformer):
    """Random Scalable and Accurate Subsequence Transform (SAST).

    RSAST [1] is based on SAST, it uses a stratified sampling strategy for subsequences selection but additionally takes into account certain 
    statistical criteria such as ANOVA, ACF, and PACF to further reduce the search space of shapelets.
    
    RSAST starts with the pre-computation of a list of weights, using ANOVA, which helps in the selection of initial points for
    subsequences. Then randomly select k time series per class, which are used with an ACF and PACF, obtaining a set of highly correlated 
    lagged values. These values are used as potential lengths for the shapelets. Lastly, with a pre-defined number of admissible starting 
    points to sample, the shapelets are extracted and used to transform the original dataset, replacing each time series by the vector of 
    its distance to each subsequence.

    Parameters
    ----------
    n_random_points: int default = 10 the number of initial random points to extract
    len_method:  string default="both" the type of statistical tool used to get the length of shapelets. "both"=ACF&PACF, "ACF"=ACF, "PACF"=PACF, "None"=Extract randomly any length from the TS
    nb_inst_per_class : int default = 10
        the number of reference time series to select per class
    seed : int, default = None
        the seed of the random generator
    classifier : sklearn compatible classifier, default = None
        if None, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used.
    n_jobs : int, default -1
        Number of threads to use for the transform.

    Reference
    ---------
    .. [1] Varela, N. R., Mbouopda, M. F., & Nguifo, E. M. (2023). RSAST: Sampling Shapelets for Time Series Classification.
    https://hal.science/hal-04311309/


    Examples
    --------
    >>> from aeon.transformations.collection.shapelet_based import RSAST
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> rsast = RSAST()
    >>> rsast.fit(X_train, y_train)
    RSAST()
    >>> X_train = rsast.transform(X_train)
    >>> X_test = rsast.transform(X_test)

    """
    
    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": False,
        "algorithm_type": "subsequence",
    }

    def __init__(
        self,
        n_random_points=10,
        len_method="both",
        nb_inst_per_class=10,
        seed=None,
        n_jobs=-1,
    ):
        super().__init__()
        self.n_random_points = n_random_points,
        self.len_method = len_method,
        self.nb_inst_per_class = nb_inst_per_class
        self.n_jobs = n_jobs
        self.seed = seed

    
    def __init__(self,n_random_points=10, nb_inst_per_class=10, len_method="both", random_state=None, classifier=None, sel_inst_wrepl=False,sel_randp_wrepl=False, half_instance=False, half_len=False,n_shapelet_samples=None ):
        super(RSAST, self).__init__()
        self.n_random_points = n_random_points
        self.nb_inst_per_class = nb_inst_per_class
        self.len_method = len_method
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state
        self.classifier = classifier
        self.cand_length_list = None
        self.kernels_ = None
        self.kernel_orig_ = None  # not z-normalized kernels
        self.kernel_permutated_ = None
        self.kernels_generators_ = None
        self.class_generators_ = None
        self.sel_inst_wrepl=sel_inst_wrepl
        self.sel_randp_wrepl=sel_randp_wrepl
        self.half_instance=half_instance
        self.half_len=half_len
        self.time_calculating_weights = None
        self.time_creating_subsequences = None
        self.time_transform_dataset = None
        self.time_classifier = None
        self.n_shapelet_samples =n_shapelet_samples

    def get_params(self, deep=True):
        return {
            'len_method': self.len_method,
            'n_random_points': self.n_random_points,
            'nb_inst_per_class': self.nb_inst_per_class,
            'sel_inst_wrepl':self.sel_inst_wrepl,
            'sel_randp_wrepl':self.sel_randp_wrepl,
            'half_instance':self.half_instance,
            'half_len':self.half_len,        
            'classifier': self.classifier,
            'cand_length_list': self.cand_length_list
        }

    def init_sast(self, X, y):
        #0- initialize variables and convert values in "y" to string
        start = time.time()
        y=np.asarray([str(x_s) for x_s in y])
        
        self.cand_length_list = {}
        self.kernel_orig_ = []
        self.kernels_generators_ = []
        self.class_generators_ = []

        list_kernels =[]
        
        
        
        n = []
        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        m_kernel = 0

        #1--calculate ANOVA per each time t throught the lenght of the TS
        for i in range (X.shape[1]):
            statistic_per_class= {}
            for c in classes:
                assert len(X[np.where(y==c)[0]][:,i])> 0, 'Time t without values in TS'

                statistic_per_class[c]=X[np.where(y==c)[0]][:,i]
                #print("statistic_per_class- i:"+str(i)+', c:'+str(c))
                #print(statistic_per_class[c].shape)


            #print('Without pd series')
            #print(statistic_per_class)

            statistic_per_class=pd.Series(statistic_per_class)
            #statistic_per_class = list(statistic_per_class.values())
            # Calculate t-statistic and p-value

            try:
                t_statistic, p_value = f_oneway(*statistic_per_class)
            except DegenerateDataWarning or ConstantInputWarning:
                p_value=np.nan
            # Interpretation of the results
            # if p_value < 0.05: " The means of the populations are significantly different."
            #print('pvalue', str(p_value))
            if np.isnan(p_value):
                n.append(0)
            else:
                n.append(1-p_value)
        end = time.time()
        self.time_calculating_weights = end-start


        #2--calculate PACF and ACF for each TS chossen in each class
        start = time.time()
        for i, c in enumerate(classes):
            X_c = X[y == c]
            if self.half_instance==True:
                cnt = np.max([X_c.shape[0]//2, 1]).astype(int)
                self.nb_inst_per_class=cnt
            else:
                cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            #set if the selection of instances is with replacement (if false it is not posible to select the same intance more than one)
            if self.sel_inst_wrepl ==False:
                choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            else:
                choosen = self.random_state.choice(X_c.shape[0], cnt)
            
            
            
            
            for rep, idx in enumerate(choosen):
                self.cand_length_list[c+","+str(idx)+","+str(rep)] = []
                non_zero_acf=[]
                if (self.len_method == "both" or self.len_method == "ACF" or self.len_method == "Max ACF") :
                #2.1-- Compute Autorrelation per object
                    acf_val, acf_confint = acf(X_c[idx], nlags=len(X_c[idx])-1,  alpha=.05)
                    prev_acf=0    
                    for j, conf in enumerate(acf_confint):

                        if(3<=j and (0 < acf_confint[j][0] <= acf_confint[j][1] or acf_confint[j][0] <= acf_confint[j][1] < 0) ):
                            #Consider just the maximum ACF value
                            if prev_acf!=0 and self.len_method == "Max ACF":
                                non_zero_acf.remove(prev_acf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_acf)
                            non_zero_acf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_acf=j        
                
                non_zero_pacf=[]
                if (self.len_method == "both" or self.len_method == "PACF" or self.len_method == "Max PACF"):
                    #2.2 Compute Partial Autorrelation per object
                    pacf_val, pacf_confint = pacf(X_c[idx], method="ols", nlags=(len(X_c[idx])//2) - 1,  alpha=.05)                
                    prev_pacf=0
                    for j, conf in enumerate(pacf_confint):

                        if(3<=j and (0 < pacf_confint[j][0] <= pacf_confint[j][1] or pacf_confint[j][0] <= pacf_confint[j][1] < 0) ):
                            #Consider just the maximum PACF value
                            if prev_pacf!=0 and self.len_method == "Max PACF":
                                non_zero_pacf.remove(prev_pacf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_pacf)
                            
                            non_zero_pacf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_pacf=j 
                            
                if (self.len_method == "all"):
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend(np.arange(3,1+ len(X_c[idx])))
                
                #2.3-- Save the maximum autocorralated lag value as shapelet lenght 
                
                if len(self.cand_length_list[c+","+str(idx)+","+str(rep)])==0:
                    #chose a random lenght using the lenght of the time series (added 1 since the range start in 0)
                    rand_value= self.random_state.choice(len(X_c[idx]), 1)[0]+1
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend([max(3,rand_value)])
                #elif len(non_zero_acf)==0:
                    #print("There is no AC in TS", idx, " of class ",c)
                #elif len(non_zero_pacf)==0:
                    #print("There is no PAC in TS", idx, " of class ",c)                 
                #else:
                    #print("There is AC and PAC in TS", idx, " of class ",c)

                #print("Kernel lenght list:",self.cand_length_list[c+","+str(idx)],"")
                 
                #remove duplicates for the list of lenghts
                self.cand_length_list[c+","+str(idx)+","+str(rep)]=list(set(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                #print("Len list:"+str(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                for max_shp_length in self.cand_length_list[c+","+str(idx)+","+str(rep)]:
                    
                    #2.4-- Choose randomly n_random_points point for a TS                
                    #2.5-- calculate the weights of probabilities for a random point in a TS
                    if sum(n) == 0 :
                        # Determine equal weights of a random point point in TS is there are no significant points
                        # print('All p values in One way ANOVA are equal to 0') 
                        weights = [1/len(n) for i in range(len(n))]
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                    else: 
                        # Determine the weights of a random point point in TS (excluding points after n-l+1)
                        weights = n / np.sum(n)
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                        
                    if self.half_len==True:
                        self.n_random_points=np.max([len(X_c[idx])//2, 1]).astype(int)
                    
                    
                    if self.n_random_points > len(X_c[idx])-max_shp_length+1 and self.sel_randp_wrepl==False:
                        #set a upper limit for the posible of number of random points when selecting without replacement
                        limit_rpoint=len(X_c[idx])-max_shp_length+1
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, limit_rpoint, p=weights, replace=self.sel_randp_wrepl)
                        #print("limit_rpoint:"+str(limit_rpoint))
                    else:
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, self.n_random_points, p=weights, replace=self.sel_randp_wrepl)
                        #print("n_random_points:"+str(self.n_random_points))
                    
                    #print("rpoints:"+str(rand_point_ts))
                    
                    for i in rand_point_ts:        
                        #2.6-- Extract the subsequence with that point
                        kernel = X_c[idx][i:i+max_shp_length].reshape(1,-1)
                        #print("kernel:"+str(kernel))
                        if m_kernel<max_shp_length:
                            m_kernel = max_shp_length            
                        list_kernels.append(kernel)
                        self.kernel_orig_.append(np.squeeze(kernel))
                        self.kernels_generators_.append(np.squeeze(X_c[idx].reshape(1,-1)))
                        self.class_generators_.append(c)
        
        print("total kernels:"+str(len(self.kernel_orig_)))
        
        if self.n_shapelet_samples!=None:
            print("Truncated to:"+str(self.n_shapelet_samples))
            
            self.kernel_permutated_ = self.random_state.permutation(self.kernel_orig_)[:self.n_shapelet_samples]
        else:
            self.kernel_permutated_ = self.kernel_orig_
        
        #3--save the calculated subsequences
        
        
        n_kernels = len (self.kernel_permutated_)
        
        
        self.kernels_ = np.full(
            (n_kernels, m_kernel), dtype=np.float32, fill_value=np.nan)
        
        for k, kernel in enumerate(self.kernel_permutated_):
            self.kernels_[k, :len(kernel)] = znormalize_array(kernel)
        
        end = time.time()
        self.time_creating_subsequences = end-start

    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_sast(X, y)

        start = time.time()
        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)
        end = time.time()
        self.transform_dataset = end-start
        
        if self.classifier is None:
            
            if X_transformed.shape[0]<=X_transformed.shape[1]: #n_features (kernels) > n_samples (intances)
                self.classifier=RidgeClassifierCV()
                print("RidgeClassifierCV:"+str("size training")+str(X_transformed.shape[0])+"<="+" kernels"+str(X_transformed.shape[1]))
            else: 
                print("LogisticRegression:"+str("size training")+str(X_transformed.shape[0])+">"+" kernels"+str(X_transformed.shape[1]))
                self.classifier=LogisticRegression()
                #self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None)

        start = time.time()
        #print('X_transformed shape')
        #print(X_transformed.shape)
        #print('X_transformed')
        #print(X_transformed)

        self.classifier.fit(X_transformed, y)  # fit the classifier
        end = time.time()
        self.time_classifier = end-start
        
        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)

