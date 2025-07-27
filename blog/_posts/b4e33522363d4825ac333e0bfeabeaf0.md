---
jupyter:
  kernelspec:
    display_name: venv
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.9
  nbformat: 4
  nbformat_minor: 5
---

::: {#0895cbba .cell .markdown}
## CO2RR 공정 datasheet 분석
:::

::: {#4e0157fd .cell .markdown}

------------------------------------------------------------------------
:::

::: {#499fd6e3 .cell .markdown}

------------------------------------------------------------------------
:::

::: {#4d576959 .cell .markdown}
Output 중 Required energy_total (MJ/kgCO), MSP 분석 예정
:::

::: {#bd5404b7 .cell .markdown}
step0. 라이브러리 정리
:::

::: {#ed16aea4 .cell .code execution_count="1"}
``` python
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from kan import KAN
from kan.utils import ex_round
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

::: {.output .stream .stdout}
    cpu
:::
:::

::: {#f34c39e8 .cell .markdown}
step1. 엑셀 파일 불러오기
:::

::: {#c0466943 .cell .code execution_count="2"}
``` python
# 엑셀 데이터 로드 
xls = pd.ExcelFile("25.01.14_CO2RR_GSA.xlsx")
df_in  = pd.read_excel(xls, sheet_name='Input')
df_out = pd.read_excel(xls, sheet_name='Output')

```
:::

::: {#12dafb3a .cell .markdown}
step2.결측값 대체 및 이상치 제거
:::

::: {#d0c919d6 .cell .code execution_count="3"}
``` python
# [기존 step2]
# df_in  = df_in.fillna(df_in.mean(numeric_only=True))
# df_out = df_out.fillna(df_out.mean(numeric_only=True))

# [수정된 step2] : 결측치가 있는 행 삭제

# 1. 삭제 전 데이터 크기 확인
print(f"결측값 삭제 전 Input 데이터 크기: {df_in.shape}")
print(f"결측값 삭제 전 Output 데이터 크기: {df_out.shape}")

# 2. 결측치가 있는 행 삭제
# 두 데이터프레임의 인덱스를 맞추기 위해, 한쪽에서 삭제된 인덱스를 다른 쪽에도 적용
na_rows_in = df_in.isnull().any(axis=1)
na_rows_out = df_out.isnull().any(axis=1)
rows_to_drop = na_rows_in | na_rows_out # 둘 중 하나라도 결측치가 있으면 삭제 대상

df_in_cleaned = df_in[~rows_to_drop]
df_out_cleaned = df_out[~rows_to_drop]

# 3. 삭제 후 데이터 크기 확인
print(f"결측값 삭제 후 Input 데이터 크기: {df_in_cleaned.shape}")
print(f"결측값 삭제 후 Output 데이터 크기: {df_out_cleaned.shape}")

# 이후 step3 에서는 df_in_cleaned와 df_out_cleaned 를 사용합니다.
# X = df_in_cleaned[[...]].values
# y = df_out_cleaned[[...]].values.reshape(-1, 1)


```

::: {.output .stream .stdout}
    결측값 삭제 전 Input 데이터 크기: (2501, 8)
    결측값 삭제 전 Output 데이터 크기: (2501, 13)
    결측값 삭제 후 Input 데이터 크기: (2501, 8)
    결측값 삭제 후 Output 데이터 크기: (2501, 13)
:::
:::

::: {#77407cf1 .cell .markdown}
이상치 제거 / 전체 데이터의 10% 이내로 극소량만 제거
:::

::: {#cb9b8719 .cell .code}
``` python
# 이상치(Outlier) 제거 (IQR 방식) ---

print("--- 이상치 제거 시작 ---")

# 1. 이상치 제거 전 데이터 개수 확인
print(f"이상치 제거 전 데이터 수: {len(df_in_cleaned)} 개")

# 2. IQR을 계산하여 이상치를 탐지하고 제거하는 함수 정의
def remove_outliers_iqr(df_in, df_out):
    # 입력 변수(X)와 출력 변수(y)를 합쳐서 전체 데이터프레임 생성
    combined_df = pd.concat([df_in, df_out], axis=1)
    
    # 이상치를 탐지할 숫자형 컬럼만 선택
    numeric_cols = combined_df.select_dtypes(include=np.number).columns
    
    # 각 컬럼에 대해 이상치 경계 계산
    Q1 = combined_df[numeric_cols].quantile(0.25)
    Q3 = combined_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 6 * IQR
    upper_bound = Q3 + 6 * IQR
    
    # 모든 컬럼에 대해 정상 범위 내에 있는 데이터만 True로 표시
    # (row의 어떤 컬럼이라도 이상치면 해당 row 전체가 False가 됨)
    condition = ~((combined_df[numeric_cols] < lower_bound) | (combined_df[numeric_cols] > upper_bound)).any(axis=1)
    
    # 정상 범위에 있는 데이터만 필터링
    df_in_no_outliers = df_in[condition]
    df_out_no_outliers = df_out[condition]
    
    return df_in_no_outliers, df_out_no_outliers

# 3. 함수를 사용하여 이상치 제거
#    이전에 결측치를 제거한 df_in_cleaned, df_out_cleaned를 사용합니다.
df_in_final, df_out_final = remove_outliers_iqr(df_in_cleaned, df_out_cleaned)

# 4. 이상치 제거 후 남은 데이터 개수 확인
removed_count = len(df_in_cleaned) - len(df_in_final)
print(f"이상치 제거 후 데이터 수: {len(df_in_final)} 개 ({removed_count} 개 제거됨)")
print("--- 이상치 제거 완료 ---\n")

# 중요: 이후 단계(데이터 분할, 정규화 등)에서는 
# 이제 'df_in_final'과 'df_out_final'을 사용해야 합니다.
# 예: X = df_in_final[...]

# 중요: 이상치가 제거된 최종 데이터프레임인 'df_in_final'과 'df_out_final'을 사용합니다.



# 입력 변수(X) 선택
# 모델이 '총 필요 에너지'를 예측하는 데 사용할 정보(컬럼)들을 선택
X = df_in_final[[
    "Current density (mA/cm2)", 
    "Faradaic efficiency (%)", 
    "CO coversion",
    "Voltage (V)", 
    "Electricity cost ($/kWh)", 
    "Membrain cost ($/m2)",
    "Catpure energy (GJ/ton)", 
    "Crossover rate"
]].values



## 이 밑의 코드 수정하기!!!! 

# 출력 변수(y) 선택
# 모델이 최종적으로 맞춰야 할 정답(타겟)을 선택
y = df_out_final['Required energy_total (MJ/kgCO)'].values.reshape(-1, 1)
```

::: {.output .stream .stdout}
    --- 이상치 제거 시작 ---
    이상치 제거 전 데이터 수: 2501 개
    이상치 제거 후 데이터 수: 2378 개 (123 개 제거됨)
    --- 이상치 제거 완료 ---
:::
:::

::: {#c1592a6a .cell .markdown}
step3. 엑셀 파일 속 data 추출 및 !predict 할 데이터 이름 입력!
:::

::: {#b7eba938 .cell .code execution_count="5"}
``` python
# 입력·출력 변수 선택 및 numpy 변환    # outlier 삭제를 원하지 않을시에는 df_in_final 을 df_in으로, df_out_final을 df_out으로 바꾸면 된다
X = df_in_final[[
    "Current density (mA/cm2)", "Faradaic efficiency (%)", "CO coversion",
    "Voltage (V)", "Electricity cost ($/kWh)",
    "Membrain cost ($/m2)", "Catpure energy (GJ/ton)",  # catpure energy? excel 파일에서 이렇게 오타가 났기에 그냥 사용
    "Crossover rate"
]].values





predicting = "MSP ($/kgCO)" # 다른 output 변수 보고싶으면 이거 보면 됨 # Required energy_total (MJ/kgCO) # MSP ($/kgCO)
###### 이거를 수정해서 다른 output도 보자

y = df_out_final[predicting].values.reshape(-1, 1)   


print(X)
print("====================")
print(y)




```

::: {.output .stream .stdout}
    [[1.97379048e+03 9.73836465e-01 3.72471012e-02 ... 2.89292283e+02
      3.14514194e+00 1.49940024e-01]
     [1.95631747e+03 9.63060776e-01 5.54118353e-02 ... 2.92153805e+02
      3.24190324e+00 2.49900040e-01]
     [1.93884446e+03 9.52285086e-01 7.35765694e-02 ... 2.95015327e+02
      3.33866453e+00 3.49860056e-01]
     ...
     [1.61155538e+02 5.37714914e-01 5.36423431e-01 ... 3.38318006e+02
      4.66133547e+00 1.65013994e+00]
     [1.43682527e+02 5.26939224e-01 5.54588165e-01 ... 3.41179528e+02
      4.75809676e+00 1.75009996e+00]
     [1.26209516e+02 5.16163535e-01 5.72752899e-01 ... 3.44041050e+02
      4.85485806e+00 1.85005998e+00]]
    ====================
    [[1.01270078]
     [0.77514288]
     [0.64809651]
     ...
     [0.78464141]
     [0.92776384]
     [1.18767663]]
:::
:::

::: {#f268c3b3 .cell .markdown}
이상치 제거
:::

::: {#39df47d1 .cell .markdown}
step4. train_set, valadation_set, test_set 만들기 (64:16:20) 우측 하단
RAW로 일단 설정해놓음 나중에 python으로 바꾸기
:::

::: {#c5e4f043 .cell .code execution_count="6"}
``` python
# 1단계: 먼저 train+val과 test로 분할 (80:20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2단계: train+val을 train과 val로 분할 (64:16, 전체 대비)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

# 최종 비율 확인
# (X[:,0])~(X[:,7]) 에 각각의 입력변수들의 값들이 각각 저장됨

print(X_val)
print(f"전체 데이터셋 크기: {len(X)}")
print(f"훈련셋 크기: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"검증셋 크기: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")  
print(f"테스트셋 크기: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")


```

::: {.output .stream .stdout}
    [[6.39764094e+02 8.75483806e-01 2.88958417e-01 ... 3.31531387e+02
      3.22830868e+00 3.92243103e-01]
     [1.92303079e+02 6.42141144e-01 1.31609356e-01 ... 3.46927896e+02
      3.07636945e+00 1.13674530e+00]
     [1.80057977e+03 5.59070372e-01 2.64424230e-01 ... 3.34950020e+02
      3.93042783e+00 1.81567373e+00]
     ...
     [1.23612555e+03 6.04132347e-01 1.88462615e-01 ... 2.99953352e+02
      4.33506597e+00 1.76129548e+00]
     [1.70637745e+03 6.59774090e-01 1.61333467e-01 ... 3.05549780e+02
      3.23470612e+00 1.97241104e+00]
     [1.03100760e+03 8.20625750e-01 1.96483407e-01 ... 3.39052379e+02
      3.72091164e+00 1.93482607e+00]]
    전체 데이터셋 크기: 2378
    훈련셋 크기: 1521 (64.0%)
    검증셋 크기: 381 (16.0%)
    테스트셋 크기: 476 (20.0%)
:::
:::

::: {#c8e1d9a9 .cell .markdown}
step4-1. 층화추출(Stratified Sampling on y)을 하고 싶을 때 step4 대신
실행 / 데이터의 밀도가 다른 점을 고려
:::

::: {#836e051a .cell .raw vscode="{\"languageId\":\"raw\"}"}
```{=ipynb}


# ===================================================================
# Step 4: 데이터 분할 (훈련/검증/테스트 6:2:2, 층화 추출 적용)
# ===================================================================

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

print("--- 데이터 분할 시작 (훈련/검증/테스트, 층화 추출 방식) ---")

# 1. 층화 추출을 위한 y값 그룹(strata) 생성
num_bins = 5
try:
    y_binned = pd.cut(y.flatten(), bins=num_bins, labels=False)
    stratify_option = y_binned
    print(f"y값을 {num_bins}개 구간으로 나누어 층화 추출을 진행합니다.")
except ValueError as e:
    print(f"경고: y값으로 층화 추출을 시도했으나 실패했습니다. ({e})")
    print("일반 무작위 추출 방식으로 전환합니다.")
    stratify_option = None

# 2. 먼저, 훈련+검증 데이터(80%)와 최종 테스트 데이터(20%)로 분할합니다.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=stratify_option
)

# 3. 다음으로, 훈련+검증 데이터(X_train_val)를 다시 훈련 데이터(75%)와 검증 데이터(25%)로 분할합니다.
#    (전체 데이터의 80% 중 75%는 60%, 25%는 20%에 해당하여, 최종적으로 60:20:20 비율이 됩니다.)

# 층화 추출을 두 번째 분할에도 일관되게 적용하기 위해 y_train_val 그룹을 다시 만듭니다.
try:
    y_train_val_binned = pd.cut(y_train_val.flatten(), bins=num_bins, labels=False)
    stratify_option_2 = y_train_val_binned
except ValueError:
    stratify_option_2 = None # 분할이 불가능하면 일반 추출

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25, # 80% 중 25%는 전체의 20%
    random_state=42,
    stratify=stratify_option_2
)

print("\n--- 최종 분할 후 데이터셋 크기 ---")
print(f"훈련 데이터 (X_train): {X_train.shape}")
print(f"검증 데이터 (X_val): {X_val.shape}")
print(f"테스트 데이터 (X_test): {X_test.shape}")
print("\n--- 데이터 분할 완료 ---")

```
:::

::: {#9cd52a27 .cell .markdown}
step5. 데이터 정규화(normalization)\_전처리 과정
:::

::: {#7c5cc170 .cell .code execution_count="7"}
``` python
# 중요: 훈련 데이터(X_train, y_train)의 최소/최대값을 기준으로 스케일러를 학습(fit)하고,
# 이 기준으로 모든 데이터셋(train, val, test)을 동일하게 변환합니다.
# 이렇게 해야 테스트 과정에서 미래 정보(테스트셋의 최소/최대값)가 모델에 유출되는 것을 막을 수 있다.
# validation dataset이나 test data로 스케일링을 할 시 데이터 누수 발생 가능

from sklearn.preprocessing import MinMaxScaler
import numpy as np



# 1. MinMaxScaler 객체 생성
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()


X_train_norm = scaler_X.fit_transform(X_train) # 훈련 데이터로 스케일러 학습 및 변환 (fit_transform)
y_train_norm = scaler_y.fit_transform(y_train) # X_train의 각 변수(컬럼)별로 최소값은 0, 최대값은 1이 되도록 변환됩니다.

# 3. 학습된 스케일러로 검증 및 테스트 데이터 변환 (transform)
# X_train의 기준으로 나머지 데이터들을 변환합니다.
X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_val_norm = scaler_y.transform(y_val)   # y_val 과 y_test 도 y_train 의 정규분포를 따라 변환된다
y_test_norm = scaler_y.transform(y_test)


#print(X_train_norm)

#print(X_val_norm)

#print(X_test_norm)

#print(y_train_norm)

#print(y_val_norm)

#print(y_test_norm)

# X_train_norm 은 [[x0~x7], [x0~x7],....,[x0~x7]] 에서 각 x0~x7은 각 열마다 각각 범위가 0~1로 범위가 변환됨
# 이 변환된 정도를 X_val_norm 과 X_test_norm도 적용받음

# 정규화 후 통계 확인
print("정규화 후 통계:")
print(f"X_train_norm: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
print(f"X_val_norm: mean={X_val_norm.mean():.4f}, std={X_val_norm.std():.4f}")
print(f"X_test_norm: mean={X_test_norm.mean():.4f}, std={X_test_norm.std():.4f}")
print(f"y_train_norm: mean={y_train_norm.mean():.4f}, std={y_train_norm.std():.4f}")
print(f"y_val_norm: mean={y_val_norm.mean():.4f}, std={y_val_norm.std():.4f}")
print(f"y_test_norm: mean={y_test_norm.mean():.4f}, std={y_test_norm.std():.4f}")

```

::: {.output .stream .stdout}
    정규화 후 통계:
    X_train_norm: mean=0.4987, std=0.2885
    X_val_norm: mean=0.5042, std=0.2888
    X_test_norm: mean=0.5027, std=0.2894
    y_train_norm: mean=0.2058, std=0.1701
    y_val_norm: mean=0.1979, std=0.1895
    y_test_norm: mean=0.2125, std=0.1781
:::
:::

::: {#9ad59c20 .cell .markdown}
step6. tensor 변환
:::

::: {#482b4c61 .cell .code execution_count="8"}
``` python
# 모든 데이터셋을 텐서로 변환 
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32, device=device)

print(f"모든 텐서가 {device}에 배치됨")
```

::: {.output .stream .stdout}
    모든 텐서가 cpu에 배치됨
:::
:::

::: {#996a7ee0 .cell .markdown}
step7. dataset dictionary 만들기
:::

::: {#d8165a42 .cell .code execution_count="9"}
``` python
dataset = {'train_input': X_train_tensor,'train_label': y_train_tensor,
            'val_input': X_val_tensor, 'val_label': y_val_tensor,
            'test_input': X_test_tensor,'test_label': y_test_tensor }

# 데이터셋 크기 확인
for key, value in dataset.items():
    print(f"{key}: {value.shape}")

#여기까지도 문제 없는듯
```

::: {.output .stream .stdout}
    train_input: torch.Size([1521, 8])
    train_label: torch.Size([1521, 1])
    val_input: torch.Size([381, 8])
    val_label: torch.Size([381, 1])
    test_input: torch.Size([476, 8])
    test_label: torch.Size([476, 1])
:::
:::

::: {#a365df2c .cell .markdown}
step8. Validation set 및 test set의 성능을 평가하는 함수 정의
:::

::: {#16def304 .cell .code execution_count="10"}
``` python
# validation set으로 성능을 평가하는 함수 만들기
def evaluate_model_performance(model, dataset, scaler_y, phase="validation"):    # phase = validation 아니면 test 이다
    
    
    if phase == "validation":
        input_tensor = dataset['val_input']
        label_tensor = dataset['val_label']
    elif phase == "test":
        input_tensor = dataset['test_input']
        label_tensor = dataset['test_label']
    else:
        raise ValueError("phase는 'validation' 또는 'test'만 가능합니다")
    
    # 예측 수행
    with torch.no_grad():
        pred_norm = model(input_tensor) #input_tensor 는 KAN이 받는 새로운 입력값인 val_inut or test_input / pred_norm은 그에 대한 출력값
    
    # 역정규화
    pred_real = scaler_y.inverse_transform(pred_norm.cpu().detach().numpy())   # pred_real 은 0~1 사이의 입력 val_input or test_input을 받고 출력된 값은 다시 역정규화 한 실제 출력값
    label_real = scaler_y.inverse_transform(label_tensor.cpu().detach().numpy()) # label_real은 0~1 사이의 입력 val_label or test_label을 받고 출력한 값 역정규화
    # inverse_transform 함수는  numpy 를 입력으로 받기 떄문에 pytorch tensor를 cpu로 옮기고 numpy로 변환
    
    
    # 성능 지표 계산 from 역정규화된 label_real, pred_real 값들 from val input or test input + numpy에서 계산
    rmse = np.sqrt(mean_squared_error(label_real, pred_real)) 
    r2 = r2_score(label_real, pred_real)  #1에 가까울수록 좋다
    mae = np.mean(np.abs(label_real - pred_real)) # 오차 절댓값 평균
    
    print(f"{phase} SET 성능 평가") # phase = validation 또는 test 
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print(f"실제값 평균: {label_real.mean():.4f}")  #label_real의 평균값(실제값)
    print(f"예측값 평균: {pred_real.mean():.4f}")   #pred_real의 평균값(KAN 모델로 예측한 값)
    
    return pred_real, label_real, {'rmse': rmse, 'r2': r2, 'mae': mae}
```
:::

::: {#7f52eab5 .cell .markdown}
step9. 1개의 KAN 모델 생성
:::

::: {#227d86c0 .cell .code execution_count="11"}
``` python
# KAN 모델 생성
model = KAN(width=[8, 12, 1], grid=3, k=3, seed=42, device=device)
```

::: {.output .stream .stdout}
    checkpoint directory created: ./model
    saving model version 0.0
:::
:::

::: {#d6debc95 .cell .markdown}
step10. KAN 학습_prune, refine
:::

::: {#947446ce .cell .code execution_count="12"}
``` python
# KAN 학습
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
model.plot()
```

::: {.output .stream .stderr}
    | train_loss: 2.26e-02 | test_loss: 1.97e-02 | reg: 3.91e+00 | : 100%|█| 50/50 [00:37<00:00,  1.32it
:::

::: {.output .stream .stdout}
    saving model version 0.1
:::

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/eaeba9bb1e3bc92d2dde7e355a7e35784b0364bd.png)
:::
:::

::: {#97aa8bd6 .cell .code execution_count="13"}
``` python
#pruning
model = model.prune(node_th=1e-2, edge_th=3e-2)  # 더 자르고 싶으면 값을 높이고, 덜 자르고 변수를 많이 있게 하고 싶으면 값을 낮추기
model.plot()
```

::: {.output .stream .stdout}
    saving model version 0.2
:::

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/57e5c77c5885b99e3cd60553c33e06355c685843.png)
:::
:::

::: {#52219276 .cell .code execution_count="14"}
``` python
#학습
model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
```

::: {.output .stream .stderr}
    | train_loss: 1.30e-02 | test_loss: 1.40e-02 | reg: 6.08e+00 | : 100%|█| 50/50 [00:14<00:00,  3.42it
:::

::: {.output .stream .stdout}
    saving model version 0.3
:::

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/6f7dabc55b8e131d8a2276875235bd5c928e31bc.png)
:::
:::

::: {#29921454 .cell .code execution_count="15"}
``` python
#refine(grid extension) 그리드 세분화(구간 세분화)(전체 구간 개수 30개로 변화)
model = model.refine(30)
model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
```

::: {.output .stream .stdout}
    saving model version 0.4
:::

::: {.output .stream .stderr}
    | train_loss: 1.06e-02 | test_loss: 1.66e-02 | reg: 6.05e+00 | : 100%|█| 50/50 [00:14<00:00,  3.43it
:::

::: {.output .stream .stdout}
    saving model version 0.5
:::

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/58188cb7fa1d25023687d2cfde0fb3c3347cabcb.png)
:::
:::

::: {#445c58e6 .cell .markdown}
step11. KAN symbolification
:::

::: {#666a6d8c .cell .code execution_count="16"}
``` python
# 자동 모드로 심볼릭 회귀 수행
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin']   #'x^(-1)', 'x^(-2)', 'x^(-3)', 'x^(-4)'
model.auto_symbolic(lib=lib)
```

::: {.output .stream .stdout}
    fixing (0,0,0) with exp, r2=0.9894718527793884, c=2
    fixing (0,0,1) with 0
    fixing (0,1,0) with x, r2=0.9779667854309082, c=1
    fixing (0,1,1) with 0
    fixing (0,2,0) with x, r2=0.6019839644432068, c=1
    fixing (0,2,1) with exp, r2=0.9966791272163391, c=2
    fixing (0,3,0) with 0
    fixing (0,3,1) with 0
    fixing (0,4,0) with 0
    fixing (0,4,1) with 0
    fixing (0,5,0) with 0
    fixing (0,5,1) with 0
    fixing (0,6,0) with x, r2=0.8695535659790039, c=1
    fixing (0,6,1) with 0
    fixing (0,7,0) with x, r2=0.7729178071022034, c=1
    fixing (0,7,1) with 0
    fixing (1,0,0) with x, r2=0.9365660548210144, c=1
    fixing (1,1,0) with x, r2=0.9798405170440674, c=1
    saving model version 0.6
:::
:::

::: {#eb7e1c7e .cell .code execution_count="17"}
``` python
# symbolic 함수로 대체 후 학습
model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
```

::: {.output .stream .stderr}
    | train_loss: 3.36e-02 | test_loss: 3.17e-02 | reg: 0.00e+00 | : 100%|█| 50/50 [00:05<00:00,  9.10it
:::

::: {.output .stream .stdout}
    saving model version 0.7
:::

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/837975d880410a0cb967aee28dd2a10b9027d46f.png)
:::
:::

::: {#a2914843 .cell .markdown}
step12. symbolic 수식 출력
:::

::: {#9988d940 .cell .code execution_count="18"}
``` python
# 심볼릭 수식 출력
formula = ex_round(model.symbolic_formula()[0][0], 4)
print("formula =" , formula)  
print('(x_1: Current density, x_2: Faradaic efficiency, ... x_8: crossover rate)')
```

::: {.output .stream .stdout}
    formula = -0.1227*x_2 - 0.0266*x_3 + 0.0293*x_7 + 0.0221*x_8 + 0.0982 + 0.6529*exp(-5.9796*x_3) + 0.2668*exp(-5.0689*x_1)
    (x_1: Current density, x_2: Faradaic efficiency, ... x_8: crossover rate)
:::
:::

::: {#f43772d9 .cell .markdown}
step13. KAN이 예측한 수식 정확도 검증을 위한 함수 정의
:::

::: {#d723e550 .cell .code execution_count="19"}
``` python
# validation set으로 성능을 평가하는 함수 만들기
def evaluate_model_performance(model, dataset, scaler_y, phase="validation"):    # phase = validation 아니면 test 이다
    
    
    if phase == "validation":
        input_tensor = dataset['val_input']
        label_tensor = dataset['val_label']
    elif phase == "test":
        input_tensor = dataset['test_input']
        label_tensor = dataset['test_label']
    else:
        raise ValueError("phase는 'validation' 또는 'test'만 가능합니다")
    
    # 예측 수행
    with torch.no_grad():
        pred_norm = model(input_tensor) #input_tensor 는 KAN이 받는 새로운 입력값인 val_inut or test_input / pred_norm은 그에 대한 출력값
    
    # 역정규화
    pred_real = scaler_y.inverse_transform(pred_norm.cpu().detach().numpy())   # pred_real 은 0~1 사이의 입력 val_input or test_input을 받고 출력된 값은 다시 역정규화 한 실제 출력값
    label_real = scaler_y.inverse_transform(label_tensor.cpu().detach().numpy()) # label_real은 0~1 사이의 입력 val_label or test_label을 받고 출력한 값 역정규화
    # inverse_transform 함수는  numpy 를 입력으로 받기 떄문에 pytorch tensor를 cpu로 옮기고 numpy로 변환
    
    
    # 성능 지표 계산 from 역정규화된 label_real, pred_real 값들 from val input or test input + numpy에서 계산
    rmse = np.sqrt(mean_squared_error(label_real, pred_real)) 
    r2 = r2_score(label_real, pred_real)  #1에 가까울수록 좋다
    mae = np.mean(np.abs(label_real - pred_real)) # 오차 절댓값 평균
    
    print(f"{phase} SET 성능 평가") # phase = validation 또는 test 
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print(f"실제값 평균: {label_real.mean():.4f}")  #label_real의 평균값(실제값)
    print(f"예측값 평균: {pred_real.mean():.4f}")   #pred_real의 평균값(KAN 모델로 예측한 값)
    
    return pred_real, label_real, {'rmse': rmse, 'r2': r2, 'mae': mae}
```
:::

::: {#fb77ff0c .cell .markdown}
Step14. Validation data(검증셋)으로 성능 평가
:::

::: {#497efffd .cell .code execution_count="20"}
``` python
# validation dataset(검증셋)으로 성능 평가 (모델 개발 과정에서)
val_pred, val_actual, val_metrics = evaluate_model_performance(model, dataset, scaler_y, "validation")

#scaler_y : step5 에서 정의한 정규화된 y 변환법
# validation --- 검증셋 이용할거라는 의미 / 나중에 
# val_pred 변수: evaluate_model_performance 함수가 반환한 첫 번째 결과물(pred_real)(모델의 예측값 배열)이 저장
# val_actual 변수: 함수가 반환한 두 번째 결과물(label_real)(실제 정답 값 배열)이 저장
# val_metrics 변수: 함수가 반환한 세 번째 결과물(RMSE, R², MAE가 담긴 딕셔너리)이 저장
```

::: {.output .stream .stdout}
    validation SET 성능 평가
    RMSE: 0.0518
    R²: 0.9616
    MAE: 0.0316
    실제값 평균: 0.4463
    예측값 평균: 0.4439
:::
:::

::: {#937e66a3 .cell .markdown}
step15. KAN이 예측한 수식의 정확도 최종계산 \_ using test_input,
test_label
:::

::: {#8acf1afc .cell .code execution_count="21"}
``` python
# 모든 모델 개발이 완료된 후 최종 한 번만 수행
print("최종 테스트셋 평가")


# 최종 테스트셋 평가(phase만 'test'로 변경하여 테스트 데이터를 사용)(이전에 정의한 evaluate_model_performance 함수를 그대로 사용)
test_pred, test_actual, test_metrics = evaluate_model_performance(
    model, dataset, scaler_y, "test"
)

# test_pred : 모델의 예측값 , test_actual : 실제 정답 값 배열 , test_metrics : (RMSE, R², MAE가 담긴 딕셔너리)

print(f"\n최종 모델 성능:")
print(f"테스트셋 RMSE: {test_metrics['rmse']:.4f}")
print(f"테스트셋 R²: {test_metrics['r2']:.4f}")
print(f"테스트셋 MAE: {test_metrics['mae']:.4f}")
```

::: {.output .stream .stdout}
    최종 테스트셋 평가
    test SET 성능 평가
    RMSE: 0.0442
    R²: 0.9683
    MAE: 0.0312
    실제값 평균: 0.4667
    예측값 평균: 0.4671

    최종 모델 성능:
    테스트셋 RMSE: 0.0442
    테스트셋 R²: 0.9683
    테스트셋 MAE: 0.0312
:::
:::

::: {#cd9bb436 .cell .markdown}
step16. KAN 예측값 / 실제값 그래프
:::

::: {#7931b4ce .cell .code execution_count="22"}
``` python
# step14. KAN 예측값 / 실제값 그래프
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))

# test_actual (x축) vs test_pred (y축)
plt.scatter(test_actual, test_pred, alpha=0.7, edgecolors='k', label='Model Predictions')

# y=x 기준선 (완벽한 예측선)
min_val = min(test_actual.min(), test_pred.min())
max_val = max(test_actual.max(), test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (y=x)')

# 그래프 제목과 축 레이블 설정
plt.xlabel("Actual " + predicting , fontsize=12)   # MSP 일떄는 MSP 로 바꾸기
plt.ylabel("Predicted " + predicting, fontsize=12) # MSP 일떄는 MSP 로 바꾸기
plt.title(f'Test Set: Actual vs. Predicted (R² = {test_metrics["r2"]:.4f})', fontsize=14)
plt.legend()
plt.grid(True)
plt.axis('equal') # x, y축 스케일을 동일하게 설정
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/843b59ea36d1e0a2667f754bc31a894d99758282.png)
:::
:::

::: {#c9f2f548 .cell .markdown}
잔차 플롯 (Residual Plot): 모델의 예측 오차 패턴 분석
:::

::: {#da62b6b9 .cell .code execution_count="23"}
``` python
# 1. 잔차 계산 (실제값 - 예측값)
residuals = test_actual - test_pred

# 2. 잔차 플롯 시각화
plt.figure(figsize=(7, 4))

# x축은 예측값, y축은 잔차
plt.scatter(test_pred, residuals, alpha=0.7, edgecolors='k')

# y=0 기준선 추가 (오차가 0인 선)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

plt.xlabel("Predicted "+predicting, fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot for Test Set', fontsize=14)
plt.grid(True)
plt.show()
```

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/dcc7592aa28cf96988ef74bb00926eb5dcce01fa.png)
:::
:::

::: {#85cd30ad .cell .markdown}
1.  개별 변수 영향도 플롯 (Partial Dependence Plot): 변수-결과 관계
    시각화
:::

::: {#ff333b84 .cell .code execution_count="24"}
``` python
import numpy as np
import matplotlib.pyplot as plt

# 1. (NameError 해결) feature_names 리스트를 먼저 정의합니다.
feature_names = [
    "Current density", "Faradaic efficiency", "CO coversion", "Voltage",
    "Electricity cost", "Membrain cost", "Capture energy", "Crossover rate"
]

try:
    print("--- 모든 입력 변수에 대한 개별 영향도 플롯 ---")
    
    # 2. (TypeError 해결) 'vars' 인자 없이 plot() 함수를 호출합니다.
    # 이렇게 하면 8개 변수 각각에 대한 영향도 그래프가 한 번에 그려집니다.
    model.plot()
    
    # 전체 그래프에 대한 제목 설정 (선택 사항)
    plt.suptitle('Partial Dependence Plots for All Features', y=1.02)
    plt.show()

except Exception as e:
    print(f"오류가 발생했습니다: {e}")
    print("이전 단계에서 'model' 객체가 성공적으로 생성되었는지 확인해주세요.")

```

::: {.output .stream .stdout}
    --- 모든 입력 변수에 대한 개별 영향도 플롯 ---
:::

::: {.output .display_data}
![](vertopal_b4e33522363d4825ac333e0bfeabeaf0/789ca991c4b884796f0a8782b7a748962e17067c.png)
:::
:::

::: {#22bee12d .cell .code execution_count="25"}
``` python
# step11의 코드를 다시 실행
formula = ex_round(model.symbolic_formula()[0][0], 4)
print("formula=", formula)

# 여기서 맨 왼쪽부터  x_1부터 x_8 
```

::: {.output .stream .stdout}
    formula= -0.1227*x_2 - 0.0266*x_3 + 0.0293*x_7 + 0.0221*x_8 + 0.0982 + 0.6529*exp(-5.9796*x_3) + 0.2668*exp(-5.0689*x_1)
:::
:::

::: {#7180d5fa .cell .markdown}

------------------------------------------------------------------------
:::

::: {#7b03ef06 .cell .markdown}

------------------------------------------------------------------------
:::

::: {#c161d08d .cell .markdown}

------------------------------------------------------------------------
:::

::: {#61f2ffb4 .cell .markdown}
step17.실제 KAN이 구현한 수식에 X값을 대입해서 결과 구해보기_역정규화
이용
:::

::: {#bd032dac .cell .code execution_count="26"}
``` python
# 이전에 학습에 사용했던 scaler_X 와 scaler_y 객체가 필요합니다.

# 1. 입력 변수(X)의 최소/최대값 추출
#    사용된 변수의 순서에 맞게 이름을 지정합니다.
feature_names = [
    "Current density (mA/cm2)", "Faradaic efficiency (%)", "CO coversion",
    "Voltage (V)", "Electricity cost ($/kWh)", "Membrain cost ($/m2)",
    "Catpure energy (GJ/ton)", "Crossover rate"
]

print("--- [입력 변수(X) 정규화 키] ---")
for i, name in enumerate(feature_names):
    min_val = scaler_X.data_min_[i]
    max_val = scaler_X.data_max_[i]
    print(f"x_{i+1} ({name}):")
    print(f"  min = {min_val:.8f}")
    print(f"  max = {max_val:.8f}\n")

# 2. 출력 변수(y)의 최소/최대값 추출
y_min = scaler_y.data_min_[0]
y_max = scaler_y.data_max_[0]

print("--- [출력 변수(y) 역정규화 키] ---")
print(f"y_min = {y_min:.8f}")
print(f"y_max = {y_max:.8f}")
```

::: {.output .stream .stdout}
    --- [입력 변수(X) 정규화 키] ---
    x_1 (Current density (mA/cm2)):
      min = 123.17073171
      max = 1999.62015194

    x_2 (Faradaic efficiency (%)):
      min = 0.50009796
      max = 0.98970612

    x_3 (CO coversion):
      min = 0.02733906
      max = 0.59988205

    x_4 (Voltage (V)):
      min = 1.30043982
      max = 3.49868053

    x_5 (Electricity cost ($/kWh)):
      min = 0.05002999
      max = 0.09999000

    x_6 (Membrain cost ($/m2)):
      min = 285.03798481
      max = 348.32067173

    x_7 (Catpure energy (GJ/ton)):
      min = 3.00039984
      max = 4.99640144

    x_8 (Crossover rate):
      min = 0.00039984
      max = 1.99880048

    --- [출력 변수(y) 역정규화 키] ---
    y_min = 0.17031000
    y_max = 1.56500635
:::
:::

::: {#b562d0ba .cell .code execution_count="27"}
``` python
# step11의 코드를 다시 실행
formula = ex_round(model.symbolic_formula()[0][0], 4)
print("formula=", formula)
print('(x_1: Current density, x_2: Faradaic efficiency, ... x_8: crossover rate)')
```

::: {.output .stream .stdout}
    formula= -0.1227*x_2 - 0.0266*x_3 + 0.0293*x_7 + 0.0221*x_8 + 0.0982 + 0.6529*exp(-5.9796*x_3) + 0.2668*exp(-5.0689*x_1)
    (x_1: Current density, x_2: Faradaic efficiency, ... x_8: crossover rate)
:::
:::

::: {#6bf9a624 .cell .markdown}
python에서도 시험삼아 몇 개 출력해보기
:::

::: {#a983cab0 .cell .code execution_count="28"}
``` python
import numpy as np
import pandas as pd
from sympy import sympify, symbols, lambdify # sympy 라이브러리 추가

# --- [자동화된 수식 처리 부분 시작] ---

# 1. KAN 모델에서 직접 심볼릭 수식 문자열을 가져옵니다.
#    이제 이 'formula_str' 변수만 있으면 모든 과정이 자동으로 처리됩니다.
#    (실제 코드에서는 model.symbolic_formula()[0][0] 로 가져오시면 됩니다)
#    예시 수식:
# formula_str = 
formula_str = model.symbolic_formula()[0][0] # 실제 모델에서 가져오는 코드

print(f"▶ 자동으로 감지된 수식: y = {formula_str}\n")

# 2. 수식에 사용될 가능성이 있는 모든 변수(x_1 ~ x_8)를 기호로 정의합니다.
#    feature_names 리스트는 이전에 정의되어 있어야 합니다.
num_features = len(feature_names)
x_vars = symbols(f'x_1:{num_features + 1}')

# 3. 문자열 수식을 실제 계산 가능한 함수로 변환합니다. (핵심 부분)
#    lambdify는 sympy 수식을 매우 빠른 numpy 함수로 바꿔주는 기능입니다.
symbolic_func = lambdify(x_vars, sympify(formula_str), 'numpy')

# 4. 새로운 '만능' 예측 함수를 정의합니다.
def auto_symbolic_prediction(x_norm):
    """
    정규화된 입력 데이터(x_norm)를 받아,
    자동으로 생성된 symbolic_func를 사용해 예측값을 반환합니다.
    """
    # x_norm 배열의 각 열을 개별 인자로 분리하여 함수에 전달합니다.
    # 예: symbolic_func(x_norm[:,0], x_norm[:,1], ..., x_norm[:,7])
    # x_norm.T는 배열을 전치하여 각 행이 변수가 되도록 합니다.
    return symbolic_func(*x_norm.T)

# --- [자동화된 수식 처리 부분 끝] ---


# 2. 역정규화에 사용되는 수식 출력 (이전과 동일)
min_val = scaler_y.min_[0]
scale_range = 1 / scaler_y.scale_[0]
print("==============================================================")
print("### 역정규화 (Inverse Normalization)에 사용되는 수식 ###")
print(f"실제값 = (정규화된 값 * (최대값 - 최소값)) + 최소값")
print(f"실제값 = (정규화된 값 * {scale_range:.4f}) + {min_val:.4f}")
print("==============================================================\n")


# 4. 테스트 데이터셋 전체를 사용하여 예측을 수행합니다.
test_input_norm = dataset['test_input']
# 여기서 새롭게 정의한 자동화 함수를 호출합니다!
y_pred_norm = auto_symbolic_prediction(test_input_norm)


# 5. 이후 모든 코드는 이전과 완전히 동일합니다.
y_pred_norm_reshaped = y_pred_norm.reshape(-1, 1)
y_pred_real_scale = scaler_y.inverse_transform(y_pred_norm_reshaped)
y_true_real_scale = scaler_y.inverse_transform(dataset['test_label'])
results_df = pd.DataFrame({
    '실제값 ': y_true_real_scale.flatten(),
    '수식 예측값 ': y_pred_real_scale.flatten()
})
pd.options.display.float_format = '{:.2f}'.format

print("--------- [단순화된 수식 모델]의 예측 성능 ---------")
print(results_df.head(30))
```

::: {.output .stream .stdout}
    ▶ 자동으로 감지된 수식: y = -0.122722534268016*x_2 - 0.0266464641334772*x_3 + 0.0293454955959183*x_7 + 0.0220815973371447*x_8 + 0.0981888482442153 + 0.652913673551474*exp(-5.97961902618408*x_3) + 0.266846907746833*exp(-5.06893444061279*x_1)

    ==============================================================
    ### 역정규화 (Inverse Normalization)에 사용되는 수식 ###
    실제값 = (정규화된 값 * (최대값 - 최소값)) + 최소값
    실제값 = (정규화된 값 * 1.3947) + -0.1221
    ==============================================================

    --------- [단순화된 수식 모델]의 예측 성능 ---------
        실제값   수식 예측값 
    0   0.38     0.35
    1   0.43     0.42
    2   0.53     0.55
    3   0.19     0.18
    4   0.60     0.61
    5   0.45     0.48
    6   0.20     0.18
    7   0.37     0.35
    8   0.46     0.45
    9   0.32     0.31
    10  0.83     0.76
    11  0.27     0.27
    12  0.35     0.34
    13  0.38     0.36
    14  0.30     0.29
    15  0.21     0.22
    16  0.31     0.26
    17  0.18     0.17
    18  0.37     0.37
    19  0.37     0.34
    20  0.90     0.70
    21  0.30     0.28
    22  0.57     0.56
    23  0.42     0.42
    24  0.29     0.32
    25  0.27     0.32
    26  0.26     0.31
    27  0.33     0.36
    28  0.37     0.35
    29  0.82     0.87
:::
:::
