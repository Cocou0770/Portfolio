import os
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from rich import print
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             matthews_corrcoef, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve)
from sklearn import metrics
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier
import lightgbm as lgbm
from xgboost import XGBClassifier
import xgboost as xgb

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
 
#각 모델의 평가 지표를 보여주는 함수
#정확도,정밀도,재현율,,F1 점수, mcc, 혼동 행렬, AUC-ROC,PR-AUC

def model_eval(model_object, model_name: str, X_test, y_test):
    # 1. 기본 예측 (임계치 0.5)
    prediction = model_object.predict(X_test)

    # 2. 최적 임계치 탐색 
    best_threshold = 0.5
    best_f1 = f1_score(y_test, prediction) 
    optimal_preds = prediction 
    y_pred_proba = None

    # predict_proba가 있는지 확인
    if hasattr(model_object, "predict_proba"):
        y_pred_proba = model_object.predict_proba(X_test)[:, 1]
        
        # 0.01부터 0.49까지 0.01 간격으로 임계치를 테스트
        thresholds = np.arange(0.01, 0.5, 0.01)
        
        for threshold in thresholds:
            threshold = round(threshold, 2)
            
            preds = (y_pred_proba >= threshold).astype(int)
            
            f1 = f1_score(y_test, preds)
            
            if (f1 > best_f1):
                best_f1 = f1
                best_threshold = threshold

        optimal_preds = (y_pred_proba >= best_threshold).astype(int)
        
        print(f"   > 최적 임계치: {best_threshold:.2f} (F1-Score: {best_f1:.3f})")
    
    else:
        print("--- (predict_proba가 없어 최적 임계치 탐색을 건너뜁니다.) ---")

    metrics_funcs = {
        '정확도 (Accuracy)': accuracy_score,
        '정밀도 (Precision)': precision_score,
        '재현율 (Recall)': recall_score,
        'F1 점수 (F1-Score)': f1_score,
        '매튜 상관 계수 (MCC)': matthews_corrcoef
    }
    
    results = {}
    
    # 임계치 변경 전 (Default)
    default_metrics = {name: func(y_test, prediction) for name, func in metrics_funcs.items()}
    results['Default (Threshold: 0.5)'] = default_metrics
    
    # 임계치 변경 후 (Optimal) - 임계값이 0.5가 아닐 때만 추가
    if best_threshold != 0.5:
        optimal_metrics = {name: func(y_test, optimal_preds) for name, func in metrics_funcs.items()}
        results[f'Optimal (Threshold: {best_threshold:.2f})'] = optimal_metrics

    results_df = pd.DataFrame(results).round(4)
    display(results_df) 
    
    
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'{model_name} 모델 평가', fontsize=18)

    # --- 혼동 행렬 (Default: 0.5) ---
    conf_matrix_default = confusion_matrix(y_test, prediction, labels=[1, 0])
    sns.heatmap(conf_matrix_default, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['폐업(1)', '운영 중(0)'],
                yticklabels=['폐업(1)', '운영 중(0)'])
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title('Confusion Matrix (Default: 0.5)', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')

    # --- 수정된 임계치가 적용된 혼동 행렬 (Optimal) ---
    if best_threshold != 0.5:
        conf_matrix_optimal = confusion_matrix(y_test, optimal_preds, labels=[1, 0])
        sns.heatmap(conf_matrix_optimal, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1],
                    xticklabels=['폐업(1)', '운영 중(0)'],
                    yticklabels=['폐업(1)', '운영 중(0)'])
        axes[0, 1].set_aspect('equal')
        axes[0, 1].set_title(f'Confusion Matrix (Optimal: {best_threshold:.2f})', fontsize=14)
        axes[0, 1].set_xlabel('Predicted Label')
        axes[0, 1].set_ylabel('True Label')
    else:
        # 최적 임계치가 0.5이거나 찾지 못한 경우
        axes[0, 1].axis('off')
        axes[0, 1].text(0.5, 0.5, 'Optimal Threshold is 0.5\n(or N/A)', 
                       ha='center', va='center', fontsize=12, wrap=True)

    # --- ROC & PR 커브 ---
    if y_pred_proba is not None:
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        axes[1, 0].set_title('ROC Curve', fontsize=14)
        axes[1, 0].legend(loc="lower right")

        # PR-AUC
        prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(rec, prec)
        axes[1, 1].plot(rec, prec, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        axes[1, 1].set_title('Precision-Recall Curve', fontsize=14)
        axes[1, 1].legend(loc="lower left")
    else:
        # predict_proba가 없는 경우
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'ROC Curve not available\n(no predict_proba)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'PR Curve not available\n(no predict_proba)', 
                       ha='center', va='center', fontsize=12)

    plt.subplots_adjust(top=0.9, hspace=0.3)
    plt.show()

    return best_threshold

class Boost_model:
    def __init__(self,model_type: str,train_data,GPU ='off',
                 early_stopping_rounds = 50,n_estimators=500,learning_rate=0.05,
                 min_split_gain=0.05, random_state=42, subsample=0.8,colsample_bytree=0.8,cv_number = 5):
        
        self.train_data = train_data
        self.model_type = model_type
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.learning_rate= learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.min_split_gain = min_split_gain
        self.scale_pos_weight = (len(self.train_data.y_train) - sum(self.train_data.y_train)) / sum(self.train_data.y_train)
        self.check_point = 0
        self.GPU = GPU
        self.cv_number = cv_number  # 튜닝 시 사용할 CV 폴드 수
        self.best_params = None # 튜닝 결과를 저장할 변수
        self.model = None
        self.pred_proba = None #테스트 결과에서 위험 점수 결과를 저장하는 변수
        self.best_threshold = None #최적의 임계값
        self.emergency_mc = None #위험 가맹점 

        if self.model_type == 'lgbm':
            self.title = "Light Gradient Boosting Machine"
        elif self.model_type == 'xgb':
            self.title = "Exreme Gradient Boosting"
        else:
             raise ValueError("정확한 부스트 모델명을 입력해주세요(lgbm,xgb)")

        default_params = {
        'n_estimators': self.n_estimators,
        'learning_rate': self.learning_rate,
        'subsample': self.subsample,
        'colsample_bytree': self.colsample_bytree,
        'random_state': self.random_state}

        self.model = self._get_model(default_params)

    def _define_search_space(self):
            if self.model_type == 'xgb':
                return {
                    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
                    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                    'max_depth': hp.quniform('max_depth', 3, 15, 1),
                    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                    'subsample': hp.uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                    'gamma': hp.uniform('gamma', 0, 0.5),
                    'reg_alpha': hp.uniform('reg_alpha', 0, 1), 
                    'reg_lambda': hp.uniform('reg_lambda', 0, 1)      
                }
                
            elif self.model_type == 'lgbm':
                return {
                    'n_estimators': hp.quniform('n_estimators', 100, 2000, 100),
                    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
                    'max_depth': hp.quniform('max_depth', 3, 15, 1),
                    'min_child_samples': hp.quniform('min_child_samples', 20, 100, 5),
                    'subsample': hp.uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': hp.uniform('reg_alpha', 0, 1), 
                    'reg_lambda': hp.uniform('reg_lambda', 0, 1) 
                }
            else:
                raise ValueError("지원하지 않는 모델 이름입니다.")

    def _get_model(self, params):
            
            model_params = params.copy() 
            
            if self.model_type == 'xgb':
                for p in ['n_estimators', 'max_depth', 'min_child_weight']:
                    if p in model_params:
                        model_params[p] = int(model_params[p])
                
                
                model_params['scale_pos_weight'] = self.scale_pos_weight
                model_params['random_state'] = self.random_state
                model_params['n_jobs'] = -1
                model_params['use_label_encoder'] = False
                model_params['early_stopping_rounds'] = self.early_stopping_rounds

                if self.GPU == 'on':
                    model_params['device'] = 'cuda'
                    model_params['tree_method'] = 'hist'

                return XGBClassifier(**model_params)

            elif self.model_type == 'lgbm':
                for p in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
                    if p in model_params:
                        model_params[p] = int(model_params[p])

                
                model_params['class_weight'] = 'balanced' 
                model_params['random_state'] = self.random_state
                model_params['n_jobs'] = -1
                model_params['verbose'] = -1

                if 'min_split_gain' not in model_params:
                     model_params['min_split_gain'] = self.min_split_gain

                if self.GPU == 'on':
                    model_params['device'] = 'cuda'

                return LGBMClassifier(**model_params)        

    def _objective(self, params):
            
            model = self._get_model(params)
            
            f1_scores_list= []
            
            skf = StratifiedKFold(n_splits=self.cv_number, shuffle=True, random_state=42)
        
            for tr_index, val_index in skf.split(self.train_data.X_train, self.train_data.y_train):
            
                X_tr, X_val = self.train_data.X_train.iloc[tr_index], self.train_data.X_train.iloc[val_index]
                y_tr, y_val = self.train_data.y_train.iloc[tr_index], self.train_data.y_train.iloc[val_index]

                if self.model_type == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
                
                elif self.model_type == 'lgbm':
                    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)],
                        callbacks=[lgbm.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)])

                preds = model.predict(X_val)
                f1_scores_list.append(f1_score(y_val, preds))
            
            score = np.mean(f1_scores_list)

            return {'loss': -score, 'status': STATUS_OK}                    

    def _tune(self, max_evals=50):
            
            print(f"--- 최적화 기준: 교차 검증 (CV={self.cv_number}) F1 점수 ---")
            space = self._define_search_space()
            trials = Trials()
        
            best_params_from_fmin = fmin(
                fn=self._objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
        
            self.best_params = best_params_from_fmin
            
            print("######## 최적의 하이퍼파라미터 ########")
            print(self.best_params)
            print("\n")   

    def fit(self):
        if self.model is None:
            raise ValueError("정확한 부스트 모델명을 입력해주세요(lgbm,xgb)")
    
        elif self.model_type == 'lgbm':
            self.model.fit(
            self.train_data.X_tr, self.train_data.y_tr,
            eval_set=[(self.train_data.X_tr, self.train_data.y_tr), (self.train_data.X_val, self.train_data.y_val)]
            ,callbacks=[lgbm.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=-1)])

        elif self.model_type == 'xgb':
            self.model.fit(
            self.train_data.X_tr, self.train_data.y_tr,
            eval_set=[(self.train_data.X_tr, self.train_data.y_tr), (self.train_data.X_val, self.train_data.y_val)], verbose=False)
        
        self.check_point += 1
        self.pred_proba = self.model.predict_proba(self.train_data.X_test)[:, 1]

    def evaluation(self):
        self.fit()
        if self.model is None:
            raise ValueError("정확한 부스트 모델명을 입력해주세요(lgbm,xgb)")
        
        self.best_threshold =  model_eval(self.model,self.title,self.train_data.X_test,self.train_data.y_test)
        optimal_preds = (self.pred_proba >= self.best_threshold).astype(int)
        self.emergency_mc = np.where(optimal_preds == 1)[0]

    def Tuner(self, max_evals=50):
            # 1. 튜닝 실행
            self._tune(max_evals=max_evals)
            
            # 2. 튜닝된 파라미터로 self.model 교체
            self.model = self._get_model(self.best_params)
            
            # 3. 평가 실행 (내부적으로 fit -> model_eval 호출)
            self.evaluation()    

    def plot_feature_importance(self,top_n=20):
        if self.check_point == 0:
            raise ValueError("모델을 먼저 학습해야 합니다")
        
        importances = self.model.feature_importances_
        features = self.train_data.X_val.columns

        idx = np.argsort(importances)[::-1][:top_n]
        plt.barh(np.array(features)[idx][::-1], np.array(importances)[idx][::-1])
        plt.title(f"{self.model_type} Feature Importance (Top {top_n})")
        plt.xlabel("Importance score")
        plt.ylabel("Features")
        plt.show()
    
    def permutation_importance_plot(self):
        if self.check_point == 0:
            raise ValueError("모델을 먼저 학습해야 합니다")

        result = permutation_importance(
            self.model, self.train_data.X_val, self.train_data.y_val,
            scoring="f1",
            n_repeats=1,
            random_state=42,
            n_jobs=-1)

        fi = pd.DataFrame({
            "feature": self.train_data.X_val.columns,
            "importance": result.importances_mean
        }).sort_values(by="importance", ascending=False)

        print(fi.head(10))

        top_features = fi.head(20)

        plt.figure(figsize=(8, 6))
        plt.barh(top_features["feature"], top_features["importance"], color="skyblue")
        plt.xlabel("Permutation Importance")
        plt.ylabel("Feature")
        plt.title(f"{self.model_type} Top 20 Features by Permutation Importance")
        plt.gca().invert_yaxis() 
        plt.show()

    def custom_threshold(self,my_threshold = 0.5): 
        custom_threshold_model_eval(self.model, self.title, self.train_data.X_test, self.train_data.y_test, my_threshold = my_threshold)  

class ShapAnalysis:
    """
    학습된 트리 기반 모델(XGBoost, LightGBM 등)의 예측 결과를
    SHAP을 이용해 분석하고 시각화하는 클래스입니다.
    """
    def __init__(self, boost_model, train_data):

        self.model = boost_model
        self.X_test = train_data.X_test
        self.y_test = train_data.y_test
        self.explainer = shap.TreeExplainer(self.model)
        
        # 분석 결과를 저장할 인스턴스 변수
        self.single_data_point = None
        self.shap_values_single = None
        self.answer = None
        self.top5_positive = None
        self.probability_class_1 = None
        self.expected_value = self.explainer.expected_value

    def select_sample(self, index=0):
        
        self.single_data_point = self.X_test.iloc[[index]]
        self.answer = self.y_test.iloc[index]

        self.shap_values_single = self.explainer.shap_values(self.single_data_point)
        print(f"--- 데이터 인덱스 {index}번 샘플에 대한 분석을 시작합니다. ---")
        return self

    def text_summary(self):

        if self.single_data_point is None:
            raise ValueError("먼저 `select_sample` 메서드를 호출하여 분석할 데이터를 선택해주세요.")

        self.probability_class_1 = self.model.predict_proba(self.single_data_point)[0, 1]
        answer_dic = {0: "정상", 1: "폐업 위기"}
        answer = answer_dic[self.answer]
        
        print("\n" + "="*60)
        print(f"가맹정 현황: {answer}({self.answer})")
        print(f"🎯 가맹점이 폐업할 예측 확률: {self.probability_class_1:.3f}\n")
        print(f"📊 모델의 평균 예측 기준값 (Base Value): {self.expected_value[0]:.3f}\n")
        

        shap_df = pd.DataFrame({
            'Feature': self.single_data_point.columns,
            'SHAP Value (기여도)': self.shap_values_single.flatten()
        })
        
        positive_shap_df = shap_df[shap_df['SHAP Value (기여도)'] > 0]
        
        positive_shap_df = positive_shap_df.sort_values(by='SHAP Value (기여도)', ascending=False)
        self.top5_positive = positive_shap_df.head(5)

        print("폐업 예측 요인 TOP 5 피쳐:")
        display(self.top5_positive)

        
        
    def LLM_summary(self):


        llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  
        )

        #표 형식을 글자로 변환
        result_string = self.top5_positive.to_markdown(index=False) # 중요도 top5
        X_test_string = self.single_data_point.to_markdown(index=False) # 가게 데이터
        table_string = result_string + X_test_string + f"가맹점의 폐업 위기 지수: {self.probability_class_1:.3f}\n"
        #프롬프트
        prompt_template = """
        당신은 위기 상황의 소상공인을 위한 최고의 비즈니스 컨설턴트입니다.
        첫번째 데이터는 특정 가계의 폐업에 영향을 미치는 요인과 그 중요도(가중치)를 나타내는 표입니다.
        두번째 데이터는 해당 가계의 매출 지표를 의미하는 데이터입니다.

        {table_data}

        ### 분석 가이드라인
        - 단순히 가중치가 높은 순서대로 요인을 나열하지 마세요
        - 가계의 데이터와 폐업 요인의 가중치를 모두 고려해서 대답하세요
        - **서로 연관된 요인들은 하나의 핵심 문제로 묶어서 종합적으로 해석해야 합니다.** 예를 들어, '최근 3개월 매출 하락'과 '최근 6개월 매출 하락' 요인이 함께 있다면, 이는 '장기적인 매출 감소 추세'라는 하나의 통합된 인사이트로 분석해야 합니다.
        - 주요 요인에 영업 개월이 있다면 영업 개월 자체가 폐업 원인이 아님을 명심하십시오
        - 주요 요인에 임대료가 있으면 가계의 매출 원가 절감을 제안하세요
        - 역세권 점수는 300m 내에 지하철 역이 있으면 +2점 300~500m 사이에 지하절 역이 있으면 +1점 입니다
        - 가계의 위치를 재배치 혹은 상권 이동등의 단시간에 비용이 많이 들어가는 조언은 자체합니다
        - 소상공인 입장에서 해결할 수 있는 해결방안을 제시합니다

        ### 당신의 과업
        1.  **핵심 위험 요약**: 표를 분석하여 가장 중요한 3가지 핵심 요인을 종합적으로 고려하고, **한 문장으로 요약**해주세요.
        2.  **솔루션 제시**: 위에서 분석한 핵심 위험을 바탕으로, 실행 가능한 **구체적인 해결 방안 3가지**를 제시해주세요.

        ### 결과 출력 형식
        결과는 다음 형식에 맞춰 작성해주세요.

        ** 가맹점 위험 요인 분석 **

        [위기 지수]: (가맹점의 폐업 위기 지수를 소수점 3자리에서 반올림)

        [핵심 위험 요약]
        (여기에 한 문장 요약)

        [솔루션 제안]
        1. **솔루션 1**: (첫 번째 해결 방안과 그에 대한 설명)
        2. **솔루션 2**: (두 번째 해결 방안과 그에 대한 설명)
        3. **솔루션 3**: (세 번째 해결 방안과 그에 대한 설명)
        """
        prompt = PromptTemplate(template=prompt_template,input_variables=["table_data"])
        chain = prompt | llm
        result = chain.invoke({"table_data": table_string})

        print('*********************가맹점 위험 요소 분석*********************')
        print(result.content)

        
    def force_plot(self):
        
        if self.single_data_point is None:
            raise ValueError("먼저 `select_sample` 메서드를 호출하여 분석할 데이터를 선택해주세요.")
        shap.initjs()
        print("\n>>>Force Plot (단일 데이터 예측 설명)")
        display(shap.force_plot(
            self.expected_value[0],
            self.shap_values_single,
            self.single_data_point
        ))

    def summary_plot(self):
        
        print("\n>>>Summary Plot (전체 특성 중요도)")
        shap_values_all = self.explainer.shap_values(self.X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_all, self.X_test, show=False)
        plt.title("SHAP Summary Plot", fontsize=14)
        plt.tight_layout()
        plt.show()

    def custom_bar_plot(self):

        if self.single_data_point is None:
            raise ValueError("먼저 `select_sample` 메서드를 호출하여 분석할 데이터를 선택해주세요.")
        
        print("\n>>>Custom Bar Plot (폐업 예측 긍정 영향 TOP 5)")
        
        top5_positive_plot = self.top5_positive.sort_values(by='SHAP Value (기여도)', ascending=True)
        colors = ['red'] * len(top5_positive_plot)
        
        plt.figure(figsize=(8, 5))
        plt.barh(top5_positive_plot['Feature'], top5_positive_plot['SHAP Value (기여도)'], color=colors)
        plt.title("폐업 예측 긍정 영향 TOP 5", fontsize=16)
        plt.xlabel("SHAP Value (기여도)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.show()

    def single_sample_analysis(self,index=0):
        self.select_sample(index=index)
        self.text_summary()
        self.LLM_summary()
        self.custom_bar_plot()
        self.force_plot()

    def LLM_analysis(self,index=0):
        self.select_sample(index=index)
        self.text_summary()
        self.LLM_summary()  
