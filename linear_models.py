from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def use_xgboost(params,train_X,train_y,X_test):
    import xgboost as xgb
    n_folds=5
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True)
    predictions=np.zeros(X_test.shape[0])
    for i,(train_index,valid_index) in enumerate(skf.split(train_X,train_y)):
        print('[Fold %d/%d]' % (i + 1, n_folds))
        X_train,X_valid=train_X[train_index],train_X[valid_index]
        y_train,y_valid=train_y[train_index],train_y[valid_index]
        d_train=xgb.DMatrix(X_train, y_train)
        d_valid=xgb.DMatrix(X_valid,y_valid)
        d_test=xgb.DMatrix(X_test)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, maximize=True,verbose_eval=100)
        print('[Fold %d/%d Prediciton:]' % (i + 1, n_folds))
        p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
        predictions+=p_test/n_folds
    return predictions

def use_lightgbm(params,train_X,train_y,X_test):
    import lightgbm as lgb
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    oof = np.zeros(train_X.shape[0])
    predictions = np.zeros(X_test.shape[0])
    for i, (train_index, valid_index) in enumerate(skf.split(train_X, train_y)):
        print('[Fold %d/%d]' % (i + 1, n_folds))
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        y_train, y_valid = train_y[train_index], train_y[valid_index]
        xg_train=lgb.Dataset(X_train,label=y_train,free_raw_data = False)
        xg_valid=lgb.Dataset(X_valid,label=y_valid,free_raw_data=False)

        clf = lgb.train(params, xg_train, 5000, valid_sets=[xg_valid], verbose_eval=50, early_stopping_rounds=50)
        oof[valid_index] = clf.predict(X_valid, num_iteration=clf.best_iteration)

        print('[Fold %d/%d Prediciton:]' % (i + 1, n_folds))
        p_test=clf.predict(X_test, num_iteration=clf.best_iteration)
        predictions += p_test / n_folds
    return predictions

DATA_DIR="c:\\Users\\tihor\\Documents\\kaggle\\santander\\"
#DATA_DIR="d:\\\\kaggle\\santander\\"
df_train=pd.read_csv(DATA_DIR+'train.csv')
df_test=pd.read_csv(DATA_DIR+'test.csv')

train_y=df_train['target'].values
train_X=df_train[[col for col in df_train.columns if col not in ['ID_code','target']]].values
del df_train
X_test=df_test[[col for col in df_test.columns if col not in ['ID_code','target']]].values
#del df_test

def run_xgb(train_X,train_y,df_test):
    params = {
            'min_child_weight': 10.0,
            'objective': 'binary:logistic',
            'max_depth': 7,
            'max_delta_step': 1.8,
            'colsample_bytree': 0.4,
            'subsample': 0.8,
            'eta': 0.025,
            'gamma': 0.65,
            'num_boost_round' : 700,
            'eval_metric':'auc',
            }
    predictions=use_xgboost(params,train_X,train_y,X_test)
    df_test['target']=predictions
    df_test=df_test[['ID_code','target']]
    df_test.to_csv("xgb_submission.csv",index=False)

def run_lgb(train_X,train_y,df_test):
    params={
            'num_leaves': 12,
            'max_bin': 119,
            'min_data_in_leaf': 10,
            'learning_rate': 0.027674233178344776,
            'min_sum_hessian_in_leaf': 0.0041135185671155455,
            'bagging_fraction': 1.0,
            'bagging_freq': 5,
            'feature_fraction': 0.05417588475081639,
            'lambda_l1': 4.960965942297994,
            'lambda_l2': 1.3793048534309067,
            'min_gain_to_split': 0.04245137968559498,
            'max_depth': 15,
            'save_binary': True,
            'seed': 1337,
            'feature_fraction_seed': 1337,
            'bagging_seed': 1337,
            'drop_seed': 1337,
            'data_random_seed': 1337,
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': 1,
            'metric': 'auc',
            'is_unbalance': True,
            'boost_from_average': False,
        }
    predictions=use_lightgbm(params,train_X,train_y,X_test)
    df_test['target']=predictions
    df_test=df_test[['ID_code','target']]
    df_test.to_csv("lgb_submission.csv",index=False)

run_lgb(train_X,train_y,df_test)