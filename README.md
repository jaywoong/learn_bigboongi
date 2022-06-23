# learn_bigboongi

> **[1유형](#idx1)** 
>
> **[2유형](#idx2)**
>
> **[단답형](#idx3)**

___



## 1유형<a id="idx1"></a>

```python
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


인덱싱
df.iloc[시작(포함):끝(포함):간격 , 열:열:간격] #index 숫자값
df.iloc[rowNumber]['colName'] #열 이름으로
df.iloc[n, m] #n행m열에 있는 값
df.iloc[1:7, 0:2] #1~7행 0~2열 dataframe
df.iloc[[n1,n2,n3], [m1,m2,m3]] #n1n2n3행 m1m2m3열 dataframe


데이터프레임 정보 확인
df.head(10)
df.tail(10)
df.shape() #행 열 개수 
df.shape[0], [1] # 행, 열 개수 
len(df) #행 개수
df.count() #결측치 아닌 행 개수
df.info() #데이터 개수, 타입
df.describe() #각 컬럼 요약 통계
df['colName'].value_counts() #특정 컬럼의 값 개수 세기
df_1 = df[(df['col_1']==1) & (df['col_2']>=30)] # 특정 케이스 추출하기

기초통계량
df['colName']. sum(), mean(), median(), max(), min(), var(), std(), mode()
df['colName'].quantile() #분위
df['colName'].unique() #고유값
df['colName'].nunique() #고유값 개수
df['colName'].skew() #왜도 Skewness
df['colName'].kurt() #첨도 Kurtosis
df['colName'].cumsum() #누적합
np.log1p('colName'), np.log(df['colName']) #로그 변환
np.sqrt(df['colName']) #제곱근 변환
np.ceil(value), floor(), trunc() #올림, 내림, 버림
round(value, 자릿수) #반올림 
abs(value) #절댓값
print("%.3f" % 변수) #소수점 출력

df.corr() #모든 변수 간 상관관계 계산하여 행렬 반환
df.corr()['colName'] #특정 컬럼에 대한 상관관계
df['colName1', 'colName2'].corr() #두 변수간의 상관관계 구할 때
df.corr(method = 'pearson' , 'spearman' , 'kendall' ) #조건 있으면 method 지정
df.cov() #모든 변수 간 공분산을 계산하여 행렬 반환
df.pct_change() #퍼센트 변화율 계산


결측치 
df.isnull().sum() #컬럼별 결측치 개수 확인
df['colName'].isnull().sum() # 특정 컬럼 결측치 개수

df.fillna(method = 'ffill' ,'bfill') #이전 값으로 대체, 이후 값으로 대체
df['colName'].fillna(df.mean()['colName']) #특정 컬럼의 평균값으로 대체
df['colName'] = df['colName'].fillna(df['분류 기준 컬럼'].map({'쟈갸': 0, '미안행': 1, '부끄러워 않고' : 2, '표현 많이 할게' : 4})) # 분류 기준별 다른 값으로 결측치 대체 (도시별 중앙값 대체 예제)

df['colName'].dropna() #axis=0 행 삭제, axis=1 열 삭제, subset='colName'
df['colName'].drop_duplicates() 


전처리 #default: axis=0, inplace=False
df['colName'].sort_values(by='colName', ascending= TF)
df['colName'].replace(대체될 값, 대체할 값)
df['colName'].replace({0 : '내가', 1 : '많이', 2 : '사랑해'}) #값에 따라 다른 값으로 대체
df['colName'].astype('타입명') #타입 변환
df['colName'] = pd.to_datetime(df['colName']) #datetime으로 변환
df['colName'].dt.year, month, day
df['rangeName'] = pd.qcut(df['colName'], q=구간 개수, labels=['구간명1', '구간명2',,, ]) #q개씩 균등하게 분할


그룹 나누기
df.group_by('colName').통계함수() 
df.group_by(['colName1', 'colName2']).mean()
a, b, c, d = df.groupby('그룹화 기준 컬럼')['colName'].median() #그룹별 값 변수로 지정
dfg = df.group_by(['colName1', 'colName2', as_index=False]) #결과 데이터프레임 뽑아 쓸땐 as_index=False 하면 편함
# 그룹 평균값으로 대체
fill_func = lambda x: x.fillna(x.mean())
df_1 = df.groupby('colName').apply(fill_func)


데이터 연결
pd.concat(['df1', 'df2']) #행 방향으로 연결
pd.merge(df1, df2') #공통된 컬럼끼리 병합 how='left right inner outer'


스케일 변환
# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['colName'] = scaler.fit_transform(df[['colName']]) #Z-score스케일 변환
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['colName'] = scaler.fit_transform(df[['colName']]) #MinMax스케일 변환
```


## 2유형<a id="idx2"></a>

```python
특성/레이블 데이터셋 나누기
import pandas as pd
data = pd.read_csv('house.csv')
X = data[data.columns[0:5]] 
y = data['colName']] 

훈련용/테스트용 데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

데이터 정규화 
1. Min-Max
from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)
X_scaled_minmax_test = scaler_minmax.transform(X_test)
2. Standard
scaler_standard = StandardScaler()
scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)

모델 학습
1. 선형회귀
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_minmax_train, y_train)
2. 로지스틱회귀
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_scaled_minmax_train, y_train)


훈련데이터에 모델 적용해 결과 예측
pred_train = model.predict(X_scaled_minmax_train)

훈련데이터의 정확도(R-square: 설명력) 확인
model.score(X_scaled_minmax_train, y_train)

훈련데이터에 모델 적용해 결과 예측
pred_test = model.predict(X_scaled_minmax_test)

테스트 데이터의 정확도(R-square: 설명력) 확인
model.score(X_scaled_minmax_test, y_test)

훈련데이터의 평가지표 상세 확인
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
```
####





## 단답형<a id="idx3"></a>



###

```
