import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression

class F1Analytics:
    '''
    Preform SVM and regression analysis on Formula 1 racing data.
    Handles train/test split, model training, and evaluation for
    a fixed data format.
    '''

    def __init__(self):
        '''
        Initialize the analytics class with datasets and configurations.
        '''
        #Make it reproducible
        self.r_seed = 18 #My lucky number

        #Load data
        self.data = pd.read_csv('F1_data.csv')
        self.data_2025 = pd.read_csv('F1_data_2025.csv')

        #SVC related
        self.svc_features = ['FastestFPLap',
                             'MeanFPLaps',
                             'StdFPLaps',
                             'DeltaBestFPLap',
                             'FasterThanTeammateFP',
                            ]
        self.svc_target = 'PointFinishRace'

        #Baysian related
        self.bayesian_fp_col = 'FasterThanTeammateFP'
        self.bayesian_race_col = 'FasterThanTeammateRace'

        #Linear regression related
        self.lin_reg_features = ['FastestFPLap',
                                'MeanFPLaps',
                                'StdFPLaps',
                                'DeltaBestFPLap',
                                'TrackTempAvgFP',
                                'AirTempAvgFP',
                                'RainAvgFP']
        self.lin_reg_target = 'FastestLapRace'

    def get_train_and_test_data(self, feature_columns, target_column):
        '''
        Splits data into train/test sets with standardization.

        Input
        ------
        feature_columns: list[string]
            Column names to use as predictors.
        target_columns: string
            Column name of target variable.

        Output
        ------
        array:
            Standardized train/test datasets
                - X_train
                - X_test
                - y_train
                - y_test
        '''
        #Remove all NaN values
        clean_data = self.data.dropna(subset = feature_columns + [target_column])

        #Input values
        X = clean_data[feature_columns]

        #Target value
        y = clean_data[target_column].astype(int)

        #Test / train split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.25, random_state = self.r_seed)

        #Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def svc_kernel_analysis(self, kernels, cs, plot = True):
        '''
        Evaluate SVC accuracy across different kernels and C values.
        These are the kernels to choose from:
        ['linear', 'rbf', 'sigmoid', 'poly'] .
        If plot = True the results gets plotted.

        Input
        ------
        kernels: list[string]
            SVC kernels types to try (e.g. ['linear', 'rbf']).
        cs: list[float]
            Penalty parameter C values to test.

        Output
        ------
        best_kernel: string
            Kernel with the highest accuracy
        best_c: float
            C with the highest accuracy
        '''

        X_train, X_test, y_train, y_test = self.get_train_and_test_data(self.svc_features, self.svc_target)

        #Test models with differnt kernels
        results = []

        for kernel in kernels:
            for c in cs:
                model = svm.SVC(kernel = kernel, C = c)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.append({'Kernel': kernel,
                                'C': c,
                                'Accuracy': accuracy
                                })

        #Data frame for plotting
        df_for_plotting = pd.DataFrame(results)

        if plot:
            #Plot accuracy against complexity for each kernel
            for kernel in df_for_plotting['Kernel'].unique():
                df = df_for_plotting[df_for_plotting['Kernel'] == kernel]
                plt.plot(df['C'], df['Accuracy'], label = kernel)

            plt.legend()
            plt.title('SVC Accuracy vs C for Different Kernels')
            plt.ylabel('Accuracy')
            plt.xlabel('Misclassification Penalty (C)')
            plt.show()

        #Find best result overall
        best_row = df_for_plotting.loc[df_for_plotting['Accuracy'].idxmax()]
        best_c = best_row['C']
        best_acc = best_row['Accuracy']
        best_kernel = best_row['Kernel']

        print(f'Highest accuracy of {best_acc:.2f} with: Kernel = {best_kernel} and C = {best_c} ')

        return best_kernel, best_c

    def train_svc(self, kernel, c, report = False, conf_matrix = False):
        '''
        Train an SVC with given kernel and C, optionally printing
        classification report and confusion matrix.

        Input
        ------
        kernel: string
            Kernel type for SVC.
        c: float
            Penalty parameter.
        report: bool
            Print classification report if True.
        conf_matrix: bool
            Plot confusion matrix if True

        Output
        ------
        model: SVC
            An sklearn SVC model
        '''
        X_train, X_test, y_train, y_test = self.get_train_and_test_data(self.svc_features, self.svc_target)

        #Training model with "best" values
        model = svm.SVC(kernel = kernel, C = c)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if report:
            print(f"------------------- Kernel: {kernel} C = {c} -------------------")
            print(classification_report(y_test, y_pred, target_names=["No Points", "Scored Points"]))

        if conf_matrix:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Points", "Scored Points"])
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix for {kernel} SVC (C = {c})")
            plt.xlabel("Predicted Result")
            plt.ylabel("Actual Result")
            cbar = disp.ax_.images[-1].colorbar
            cbar.set_label("Number of Drivers")
            plt.show()

        return model


    def __beta_post(self, success, n_trails, alpha_prior, beta_prior, n_theta = 1000):
            '''
            Compute beta posterior distribution and 95% credible interval

            Input
            ------
            success: int
                Number of successes observed.
            trails: int
                Total number of trials.
            alpha_prior: int
                Prior alpha parameter.
            beta_prior: int
                Prior beta parameter.
            n_theta: int
                Resolution for the PDF array

            Output
            ------
            ci_lower: float
                Lower bound of posterior credible interval.
            ci_upper: float
                Upper bound of posterior credible interval.
            post_pdf: array
                Posterior probability density function evaluated at n_points.
            alpha_post: float
                Posterior alpha parameter.
            beta_post: float
                Posterior beta parameter.
            post_mean: float
                Posterior mean of the Beta distribution.
            '''
            alpha_post = alpha_prior + success
            beta_post = beta_prior + (n_trails - success)
            post_mean = alpha_post / (alpha_post + beta_post)

            ci_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
            ci_upper = stats.beta.ppf(0.975, alpha_post, beta_post)

            theta = np.linspace(0, 1, n_theta)
            post_pdf = stats.beta.pdf(theta, alpha_post, beta_post)

            return ci_lower, ci_upper, post_pdf, alpha_post, beta_post, post_mean

    def get_posterior(self, alpha_prior, beta_prior, n_theta = 1000):
        '''
        Calculate Bayesian posterior for probability that a driver is faster in the race
        given their FP performance (faster vs slower).

        Input
        ------
        alpha_prior: int
            Prior alpha parameter.
        beta_prior: int
            Prior beta parameter.
        n_theta: int
            Number of points in the posterior PDF

        Output
        ------
        dictionary with:
            - Lower bounds of the credible intervals (list[float])
            - Upper bounds of the credible intervals (list[float])
            - Posterior PDF (list[array])
            - Posterior mean (list[float])
            - Posterior alpha parameters (list[float])
            - Posterior beta parameters (list[float])
        '''

        #Drop NaN values
        data_bayesian = self.data[[self.bayesian_fp_col, self.bayesian_race_col]].dropna()

        #Convert to int
        fp = data_bayesian[self.bayesian_fp_col].astype(int)
        race = data_bayesian[self.bayesian_race_col].astype(int)

        #Faster in race given faster in FP
        idx_fast_fp = (fp == 1)
        n_fast_race_fast_fp = race[idx_fast_fp].sum()
        n_fast_fp = fp.sum()

        #Faster in race given slower in FP
        idx_slow_fp = (fp == 0)
        n_fast_race_slow_fp = race[idx_slow_fp].sum()
        n_slow_fp = fp.sum()

        ci_l_slow, ci_u_slow, pdf_slow, alpha_slow, beta_slow, mean_slow = self.__beta_post(
                                                                n_fast_race_slow_fp,
                                                                n_slow_fp, alpha_prior,
                                                                beta_prior, n_theta)
        ci_l_fast, ci_u_fast, pdf_fast, alpha_fast, beta_fast, mean_fast = self.__beta_post(
                                                                n_fast_race_fast_fp,
                                                                n_fast_fp, alpha_prior,
                                                                beta_prior, n_theta)

        return {
            'ci_lower': [ci_l_slow, ci_l_fast],
            'ci_upper': [ci_u_slow, ci_u_fast],
            'post_pdf': [pdf_slow, pdf_fast],
            'post_mean': [mean_slow, mean_fast],
            'alpha_post': [alpha_slow, alpha_fast],
            'beta_post': [beta_slow, beta_fast]
        }

    def train_lin_reg(self, model_name = 'ElasticNet', l1_ratio = 0.5, poly_order = 1, scores = True):
        '''
        Train a linear regression model with cross-validation.

        Input
        ------
        model_name: string
            Name of model one of {'ElasticNet','Lasso','Ridge','LinearRegression'}.
        l1_ratio: float
            L1 ratio for ElasticNet (ignored otherwise).
        poly_order: int
            Polynomial feature order.
        scores: bool
            If True, print tuning results.

        Output
        ------
        best_model: Pipeline
            Fitted sklearn pipeline
        '''
        #Prepare data
        data_fastest_lap_pred = self.data.dropna(subset = self.lin_reg_features + [self.lin_reg_target])
        #Input values
        X = data_fastest_lap_pred[self.lin_reg_features]

        #Output values
        y = data_fastest_lap_pred[self.lin_reg_target]

        #Group for Cross Validation so that GPs are not split
        groups = data_fastest_lap_pred['GP']

        #Choose model
        model = {'ElasticNet': ElasticNet,
                  'Lasso': Lasso,
                  'Ridge': Ridge,
                  'LinearRegression': LinearRegression}[model_name]

        #Cross validation setup
        alphas = np.logspace(-4, 4, 100)
        cv = GroupKFold(n_splits = 5, shuffle = True, random_state = self.r_seed)

        print(f'------------------- Model: {model_name} ------------------- ')
        #Create pipeline
        if model_name == 'ElasticNet':
            pipeline = Pipeline([("scaler", StandardScaler()),
                                ("poly", PolynomialFeatures(degree = poly_order, include_bias=False)),
                                ("model", model(max_iter = 50000, l1_ratio = l1_ratio))])
        elif model_name == 'LinearRegression':
            pipeline = Pipeline([("scaler", StandardScaler()),
                                ("poly", PolynomialFeatures(degree = poly_order, include_bias=False)),
                                ("model", model())])
        else:
            pipeline = Pipeline([("scaler", StandardScaler()),
                                ("poly", PolynomialFeatures(degree = poly_order, include_bias=False)),
                                ("model", model(max_iter = 50000))])

        if model_name == 'LinearRegression':
            param_grid = {}
        else:
            param_grid =  {'model__alpha': alphas}

        #Grid search for best alpha
        grid = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = GroupKFold(n_splits = 5, shuffle = True, random_state = self.r_seed),
        scoring = 'neg_mean_absolute_error',
        n_jobs = -1
        )

        #Fit model
        grid.fit(X, y, groups = groups)

        #Evaluate with best alpha
        best_model = grid.best_estimator_

        if scores:
        #MAE and R^2 scores
            mae_scores = -cross_val_score(best_model, X, y, cv = cv, groups = groups, scoring = 'neg_mean_absolute_error')
            r2_scores  =  cross_val_score(best_model, X, y, cv = cv, groups = groups, scoring = 'r2')

            #Print out the best alpha
            if model_name != 'LinearRegression':
                print(f"Best α: {grid.best_params_['model__alpha']:.3f}")
            print(f"Best MAE: {-grid.best_score_:.3f}")

            print(f"MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
            print(f"R²:  {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

        return best_model

    def get_most_influential_features(self, model):
        '''
        Indentify and plot most influential features from a regression
        pipeline.

        Input
        ------
        model: Pipeline
            Fitted sklearn pipeline.

        Output
        ------
        coef_df: DataFrame
            DataFrame of features sorted by absolute coefficient magnitude.

        '''
        #Extract coefficients and corresponding feature names
        reg_model = model.named_steps['model']
        poly  = model.named_steps['poly']

        coef_df = pd.DataFrame({
            "Feature": poly.get_feature_names_out(self.lin_reg_features),
            "Coefficient": reg_model.coef_
        })

        # Sort by absolute importance
        coef_df["AbsValue"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values(by="AbsValue", ascending=False)

        # --- Plot ---
        sns.barplot(
            data=coef_df,
            x="AbsValue", y="Feature",
            palette="viridis",
            hue = "Feature"
        )

        plt.title("Most Influential Features")
        plt.xlabel("Coefficient Magnitude")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

        return coef_df.reset_index(drop = True)

    def test_lin_reg_model(self, model, own_data = False, csv_file = ''):
        '''
        Test the regression model on new data and plot prediction errors
        per Grand Prix.

        Input
        ------
        model: Pipeline
            Fitted sklearn pipeline.
        own_data: bool
            If true user can add a csv file with data.
        csv_file: string
            Path to dataset

        Output
        ------
        results_df: DataFrame
            DataFrame with
                - GP name
                - Predicted fastest laptime
                - Actual fastest laptime
                - Error
                - Sign
        '''
        if own_data == False:
            #Load the 2025 dataset
            data = self.data_2025
        else:
            try:
                data = pd.read_csv(csv_file)
            except Exception as e:
                print(f'Failed to load .csv file. Using 2025 data.\nError: {e}')
                data = self.data_2025

        # Ensure no missing input features
        data = data.dropna(subset = self.lin_reg_features + [self.lin_reg_target])

        # Collect predictions for all races
        results = []

        for gp_name in data["GP"].unique():
            data_gp = data[data["GP"] == gp_name].copy()
            if data_gp.empty:
                continue

            # Ensure same columns and order as training
            X_gp = data_gp[self.lin_reg_features]
            y_real = data_gp[self.lin_reg_target]

            # Predict using tuned Ridge pipeline
            y_pred = model.predict(X_gp)

            # Take the fastest (minimum) predicted and real laps
            predicted = np.min(y_pred)
            actual = np.min(y_real)
            error = predicted - actual

            results.append({
                "GP": gp_name,
                "Predicted": predicted,
                "Actual": actual,
                "Error": error
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        results_df["Sign"] = results_df["Error"].apply(lambda x: x > 0)

        sns.barplot(x=results_df['Error'], y=results_df['GP'], palette='seismic', hue=results_df['Sign'], legend=False)
        plt.axvline(0, color='black')
        plt.xlabel("Prediction Error (s)")
        plt.ylabel("Grand Prix")
        plt.title("Error per Grand Prix")
        plt.show()

        return results_df