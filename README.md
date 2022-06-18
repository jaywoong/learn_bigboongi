# learn_bigboongi

> **[1유형](#idx1)** 
>
> **[2유형](#idx2) **
>
> **[단답형](#idx3)**

___



## 1유형<a id="idx1"></a>

```python
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

인덱싱
df.iloc[첫 행 :  끝 행 : 간격 , 열 : 열 : 간격] #다 숫자
df.iloc[rowNumber]['colName'] #열 이름으로

데이터프레임 정보 확인
df.head(10)
df.tail(10)
df.shape
df.info()
df.describe() #이것저것 다 보기 
df['colName'].unique() #그룹별 값 구할 때 참고
df['colName'].value_counts() #값별 개수 세기
df['colName'].isnull().sum() #결측치 개수 세기

기초통계량
df['colName'].mean(), median(), max(), min(), std(), sum()
df['colName'].quantile(0.75) #사분위수
round(value), abs(value)

상관관계
df.corr() #str 제외하고 correlation matrix 생성
df['colName1', 'colName2'].corr() #두 변수간의 상관관계 구할 때
df.corr(method = 'pearson' , 'spearman' , 'kendall' ) #조건 있으면 method 지정

전처리 # axis=0, inplace=False (default) 
df['colName'].fillna(대체할 값)
df['colName'].fillna(대체할 값) 
df.fillna(method = 'ffill' ,'bfill') #이전 값 대체, 이후 값 대체
df['colName'] = df['colName'].fillna(df['분류 기준 컬럼'].map({'미오': 0, '내가': 1, '많이' : 2, '사랑해' : 4})) # 분류 기분별 다른 값으로 결측치 대체 (도시별 중앙값 대체 예제)
df['colName'].dropna()
df['colName'].replace(대체될 값, 대체할 값)
df['colName'].replace({0 : '미오', 1 : '사랑해'}) #값에 따라 다른 값으로 대체
df['colName'].drop_duplicates()
df['colname'].astype(형태) #컬럼 데이터 타입 변경
df['colname'].sort_values(ascending= True | False)
df.group_by('colName').mean()
a, b, c, d = df.groupby('그룹화 기준 컬럼')['colName'].median() #그룹별 값 변수로 지정
dfg = df.group_by(['colName1', 'colName2', as_index=False]) #결과 데이터프레임 뽑아 쓸땐 as_index=False 하면 편함

```



## 2유형<a id="idx2"></a>







## 단답형<a id="idx3"></a>



### 