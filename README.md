# learn_bigboongi

> **[1유형](#idx1)** 
>
> **[2유형](#idx2)**
>
> **[단답형](#idx3)**


___




# 1유형<a id="idx1"></a>

```python
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

row=행, column컬럼=열

데이터셋 가져오기
import pandas as pd
import numpy as np
df= pd.read_csv('데이터셋 위치경로')


정보 확인
df.shape # (행 개수, 열 개수) df.shape[0] 행 개수, df.shape[1] 열 개수
len(df) # 결측치 포함한 행 개수
df.count() # 결측치 제외한 행 개수
df['열이름'].value_counts() # 해당 열의 데이터별 개수
df.info() # 열 이름, 개수, 데이터 타입
df.describe() # 열별 요약 통계(개수, 평균, 표준편차 등)
df['열이름'].unique() #고유값
df['열이름'].nunique() #고유값 개수
df_1 = df[(df['col_1']==1) & (df['col_2']>=30)] # 특정 조건인 데이터만 추출해 데이터프레임 반환
s= df[df['city']=='서울']['f1'].median()
series.index(순서) # 시리즈의 인덱스이름 뽑기
df.columns[순서] # 데이터프레임의 열이름 뽑기


인덱싱
iloc: 행/열순서(숫자), loc: 행/열이름
df.iloc[시작(포함):끝(포함):간격 , 열:열:간격] #인덱스가 숫자
df.iloc[1, 2] #2행 3열에 있는 값
df.iloc[1:7, 0:2] #1~7행 0~2열 DataFrame반환
df.iloc[[n1,n2,n3], [m1,m2,m3]] #n1n2n3행 m1m2m3열 DataFrame반환
df.iloc[rowNumber]['colName'] #열이름
df.loc['a':'b", 'c':'d'] #행이름a~b 열이름c~d DataFrame. 인덱스가 문자


기초통계량
df['열이름']. sum(), mean(), median(), max(), min(), var(), std(), mode()
df['열이름'].quantile() #분위

df[(df['age']- ★np.floor(df['age']))!= 0] # 정수가 아닌 데이터 찾기
np.ceil(value), floor(), trunc() #올림, 내림, 버림
round(value, 자릿수) #반올림 

df['열이름'].skew() #왜도 Skewness
df['열이름'].kurt() #첨도 Kurtosis
df['열이름'].cumsum() #누적합
np.log1p('열이름'), np.log(df['열이름']) #로그 변환
np.sqrt(df['열이름']) #제곱근 변환
abs(value) #절댓값
print("%.3f" % 변수) #소수점 출력


이상치: IQR 범위 밖의 값
Q1= df['열이름'].quantile(0.25)
Q3= df['열이름'].quantile(0.75)
min= Q1 - 1.5*IQR 
max= Q1 + 1.5*IQR 
outlier= df[(df['열이름']<min) | (df['열이름']>max)]

df.corr() # 모든 변수 간 상관관계 계산하여 행렬 반환
df.corr()['열이름'] # 특정 열에 대한 상관관계
df['열이름1', '열이름2'].corr() # 두 변수간의 상관관계 구할 때
df.corr(method = 'pearson' , 'spearman' , 'kendall' ) # 조건 있으면 method 지정
df.cov() # 모든 변수 간 공분산을 계산하여 행렬 반환
df.pct_change() # 퍼센트 변화율 계산


결측치 
df.isnull().sum() # 열별 결측치 개수
df['열이름'].isnull().sum() # 특정 열 결측치 개수
df.count(), df.shape[0] # 열별 데이터 개수

df['열이름'].dropna() # axis=0 행 삭제, axis=1 열 삭제
df.dropna(★subset=['열이름'], inplace=True) # 특정 열의 결측치 행 삭제

df.fillna(method= 'ffill','bfill') # 결측치를 ffill이전 값으로 대체, bfill이후 값으로 대체
df['열이름'].fillna(df.mean()['열이름']) # 결측치를 특정 열의 평균값으로 대체

df['열이름']= df['열이름'].fillna(df['분류 기준 열'].map({'쟈갸': 0, '고마워': 1, '많이' : 2, '반성해' : 4})) # 분류 기준별 다른 값으로 결측치 대체 (도시별 중앙값 대체 예제)
df['f1']= df[(df['city']=='서울')]['f1'].fillna(a) 
df['f1'].fillna(df['city'].map({'서울':s,'경기':k,'부산':b,'대구':d}))



전처리 
디폴트: axis=0(행), inplace=False(원본 변경 안함)
df.sort_values(by='정렬기준 열', ascending=True, inplace=False) # T오름차순, F내림차순
df.sort_values(by=['열1','열2']) # 열1 기준 정렬 후 같은 값은 열2 기준 
df['열이름'].sort_values() # 해당 열만 정렬(series)
df.sort_index(by='정렬기준 행', ascending= TF) 

df.drop(axis=0행/1열, index='행이름', ★columns='열이름', inplace=True) # 행/열 삭제
df.drop(df[df['age']-np.floor(df['age'])!=0].index★, inplace=True) # 조건에 해당하는 행 삭제
df.drop_duplicates(subset='열이름') # 내용이 중복되는 행 제거

df['열이름'].replace(대체될 값, 대체할 값)
df['열이름'].replace({0: '내가', 1: '많이', 2: '미안해'}) # 값에 따라 다른 값으로 대체

df['열이름'].astype('타입명') # 타입 변환


df['열이름']= ★pd.to_datetime( ★df['열이름']) # datetime으로 타입 변환.
df['열이름'].dt.year,month,day # 연도,월,일을 반환
df['열이름'].dt.dayofweek # 월요일~일요일을 0~6으로 반환
df.resample(rule, axis=0) # Datetime Index를 원하는 주기로 나눔. rule='W' 1주, '2W' 2주, 'M' 월

df['새로 생성할 열이름']= ★pd.qcut(df['기준 열이름'], ★q=구간 개수, ★labels=['구간명1', '구간명2',,,]) # q개씩 균등하게 분할


그룹 나누기
df.groupby('그룹기준 열')['계산할 열'].통계함수() 
df.groupby('이름')['휴일'].sum() # 인당 남은 휴가의 총합
df.groupby('월')['휴일'].max() # 월별 남은 휴가의 최대
df.groupby(['월', '이름']).agg({'휴일':'mean', '야근':'sum'}) # 월별 인당 휴일 평균, 야근 총합
df.groupby(★['열이름1', '열이름2'])★[['열이름3']].mean() #열1,2별 열3 평균

a,b,c,d= df.groupby('city')['f1'].median() # 같은 열(city)의 특정 열(f1) 중앙값
df= df.group_by(['열이름1', '열이름2', as_index=False]) # 데이터프레임으로 결과 뽑을땐 as_index=False

fill_func= lambda x: x.fillna(x.mean())
df_1= df.groupby('열이름').apply(fill_func) # 그룹 평균값으로 대체


데이터 연결
★pd.concat( ★[df1, df2], axis=0) # axis=0 행 방향(밑에 붙임), axis=1 열 방향(옆에 붙임)
★pd.merge( ★df_left, df_right, how='inner', on='병합 기준 열') # 공통된 열끼리 병합. NaN값이 적은 df를 기준으로 병합 how=left, right, inner, outer


행열 이름 지정
df.set_index() # 인덱스명 부여
df.reset_index() # 인덱스명 제거, 행번호(0~)부여
df.set_index('열이름', drop=True, append=True, inplace=True) # 해당 열을 인덱스명으로 씀, 인덱스로 세팅한 열을 삭제할건가, 기존에 있던 인덱스는 유지할건가 
df.reset_index(drop=True, inplace=True) 



스케일 변환
* StandardScaler # Z-score스케일 변환
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['열이름] = scaler.fit_transform(df★[['열이름']]) # 괄호 2번 !!!

* MinMaxScaler # MinMax스케일 변환
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['열이름'] = scaler.fit_transform(df[['열이름']])

```



# 2유형<a id="idx2"></a>

```python
1. 데이터 나누기
X = pd.read_csv("X_train.csv") # 훈련용
y = pd.read_csv("y_train.csv") # 훈련용
t = pd.read_csv("X_test.csv") # 테스트용
연속형 or 범주형 변수 확인 head(), info()

훈련용 데이터
X_num = X[['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Pri
X_cat = X[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'
X_cat = pd.get_dummies(X_cat)

테스트용 데이터
t_num = t[['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Pri
t_cat = t[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'
t_cat = pd.get_dummies(t_cat)

원핫인코딩한 범주형 변수들 합치기
X_cat, t_cat = X_cat.align(t_cat, join='inner', axis=1)


2. 연속형 변수 스케일링
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(X_num) # fit은 훈련용으로!!
X_scaled = minmax.transform(X_num)
t_scaled = minmax.transform(t_num)


3. 최종 데이터 (연속형+범주형)
X_train = pd.concat([ pd.DataFrame(X_scaled), X_cat ], axis=1) #데이터프레임을 붙
X_test = pd.concat([ pd.DataFrame(t_scaled), t_cat ], axis=1)
y_train = y['Reached.on.Time_Y.N']


4. 모델 적용 (훈련용 데이터)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train) #fit은 훈련용으로!!!


5. 결과 예측 (테스트용 데이터)
pred_val = model.predict_proba(X_test)[:, 1] #참인 확률 값만 가져오기 (0:False, 1:True)
pred_df = pd.DataFrame(pred, columns=['pred_df']) #데이터프레임으로 만들기
pred_final = pd.concat([test['ID'], pred_df], axis=1) #붙이기


6. 제출
pred_final.to_csv("20220625.csv", index=False) #인덱스 빼기
```


```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *

1. 연속형/범주형 변수 구분
X_train.info()
X_train.head()
X_train = X_train.drop(['Name','Ticket','Cabin','Fare','Embarked'], axis=1) # 불필요한 열 제거
X_test = X_test.drop(['Name','Ticket','Cabin','Fare','Embarked'], axis=1)

2. 범주형 변수 처리
X_train['범주형변수'].nunique()
X_train.isnull().sum() # df['col'].fillna() or df = df.drop('col')

2-1. nunique() 20이하
: 원핫인코딩 → 각 데이터별로 새로운 열 생성 (Sex→ Sex_Female, Sex_Male, 데이터는 0,1)
X_train_onehot = pd.get_dummies(df_train['숫자 아닌 범주형변수 Sex'])
X_test_onehot = pd.get_dummies(X_test['Sex'])
X_train_concat = pd.concat([X_train, X_train_onehot], axis=1)
X_train_concat.drop(['Sex'], axis=1, inplace=True)


2-2. unique() 20이상(100이상은 열drop) 
: 라벨인코딩 → 데이터를 0,1,2,3,,로 라벨링. concat 필요없음
le = LabelEncoder()
le = le.fit(X_train['col'])   #X_train['col']을 fit
X_train['col'] = le.transform(X_train['col'])   # X_train['col']에 따라 encoding
X_test['col'] = le.atransform(X_test['col'])   # X_test['col']에 따라 encoding

#ID 컬럼 추출 후 제거


3. 연속형 변수 스케일링
MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train) #fit은 학습데이터로 해야됨
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

결과 안 좋으면 StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler_standard.transform(X_train)
X_test_scaled = scaler_standard.transform(X_test)


(+ 모델 성능 확인)
from sklearn.model_selection import train_test_split
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, random_state=200)

from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(max_depth = 10, random_state = 5)
rf.fit(X_train_val, y_train_val)
pred_val = rf.predict_proba(X_test_val)[:,1]

from sklearn.metrics import roc_auc_score        
print(roc_auc_score(y_test_val, pred_val))


4. 모델 생성
4-1. 분류 :  y가 범주형 변수
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 10, random_state = 5)
rf.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score  # 분류 검사
print(roc_auc_score(y_test_m, pred))


4-2. 회귀 :  y가 연속형 변수
from sklearn.ensemble import RandomForestRegressor  
rf = RandomForestRegressor(max_depth = 10, random_state = 5)
rf.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error   # 회귀 검사
print(mean_squared_error(y_test_m, pred))


5. 예측 : 답 제출 형식(문제에서 줌)이 결정 
5-1. predict : 결과값. 이진분류(참/거짓 어디에 해당), 다중분류(어느 시장/범위에 속할지 예측해라)
pred = rf.predict(X_test)   #결과값  (이진분류 참, 거짓 반환) (다중분류 집단 번호 반환)


5-2. predict_proba: 확률값. 이진분류(참/거짓일 확률), 다중분류(특정 시장/범위에 속할 확률을 예측해라)
pred = rf.predict_proba(X_test)[:, 1]   # 확률값 (이진분류0=거짓, 1=참) (다중분류 각 집단의 순서에 맞게)


6. 답 제출
answer= pd.DataFrame({ 'PassengerId': X_test_concat.PassengerId, 'Survived': pred }) # 데이터프레임생성, { '열이름1': 값1, '열이름2': 값2, , ,'열이름8': 값8 }  
answer.to_csv('003000000.csv', index=False) # 데이터프레임 인덱스 삭제하고 제출



모델 점수 낮으면
1. train_test_split에 random_state 바꾸기 
2. RandomForestClassifier의 max_depth 바꾸기
3. 결측치가 많거나 클래스가 다양한 컬럼 제거
4. MinMaxScaler -> StandardScaler 바꾸기 
5. Encoder 바꾸기

```

####
범주형 변수 분류 예측 구분하기~  또 틀리면 묭청이

![image](https://user-images.githubusercontent.com/85271084/204086247-e8181b71-57ca-4ee3-92ea-46829a5d0c3e.png)


```python
데이터 정규화 - 연속형
1. Min-Max
from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train) #fit은 학습데이터로 해야됨
X_scaled_minmax_train = scaler_minmax.transform(X_train)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

2. Standardization
scaler_standard = StandardScaler()
scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)


모델 학습
1. 선형 회귀
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_minmax_train, y_train)

2. 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_scaled_minmax_train, y_train)

3. 랜덤포레스트
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_scaled_train, y_train)


정확도 확인
1. R-square 설명력
model.score(X_scaled_minmax_train, y_train) #훈련데이터
model.score(X_scaled_minmax_test, y_test) #테스트데이터

2. RMSE
import numpy as np
from sklearn.metrics import mean_squared_error 
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련데이터 RMSE:", np.sqrt(MSE_train))
print("테스트데이터 RMSE:", np.sqrt(MSE_test)

3. 상세 평가지표
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)


교차검증
1. cross_val_score : 랜덤 없음
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print("5개 테스트 셋 정확도:", scores)
print("정확도 평균:", scores.mean())

2. KFold : 랜덤 있음
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(model, X_train, y_train, cv=kfold)
print("5개 폴드의 정확도:", scores)

3. ShuffleSplit : 임의 분할
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=0.5, train_size=0.5, random_state=42)
score = cross_val_score(model, X_train, y_train, cv=shuffle_split)
print("교차검증 정확도:", scores)

```


```python
판다스
import pandas as pd
print(dir(pd)) # dir을 통해 사용 가능한 함수 확인
print(dir(pd.DataFrame)) # 데이터프레임에서 할 수 있는 것들은?
print(help(pd.DataFrame.drop)) # 데이터프레임에서 결측치 drop을 어떻게 사용했더라?

사이킷런
import sklearn
print(sklearn.__all__)
print(sklearn.preprocessing.__all__) # 전처리 무엇을 할 수 있지?
print(help(sklearn.preprocessing.MinMaxScaler)) # 민맥스스케일 어떻게 사용하지?
print(help(sklearn.ensemble.RandomForestClassifier())) # 랜덤포레스트 어떻게 썻더라? 

▶ 해당 출력물을 메모장에 복사한 뒤 검색 기능을 활용에 문서 활용
```




# 단답형<a id="idx3"></a>

##

```
