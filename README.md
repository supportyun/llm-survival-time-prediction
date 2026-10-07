# LLM-Based Survival-Time Prediction with Censoring-Adjusted Targets

우측 검열이 있는 합성 임상자료에서 보정된 생존시간 목표값을 구성하고, 자연어로 표현된 환자 특성으로부터 기대 생존시간을 예측하도록 Llama를 미세조정한 소규모 탐색 연구입니다.

> **Project status:** Completed exploratory project (Sep 2025–Feb 2026)

## Project Overview

이 저장소는 2025년 9월부터 2026년 2월까지 수행한 탐색 연구의 코드와 결과를 정리한 포트폴리오입니다. 연구 질문은 생존곡선 전체가 아니라 검열정보를 반영해 구성한 환자별 기대 생존시간 하나를 LLM으로 예측할 수 있는지에 한정했습니다.

## Research Question

대규모 언어모델이 환자의 기초 특성을 설명한 짧은 임상 문장을 입력받아, 우측 검열을 고려해 구성한 기대 생존시간을 숫자로 출력하도록 학습할 수 있는지 확인했습니다.

이 프로젝트의 예측 대상은 생존곡선 전체가 아니라 환자별 기대 생존시간 하나입니다.

$$
\text{patient description}(X_i)
\longrightarrow
\widehat{E[T_i \mid X_i]}
$$

## Data

- PBC 자료의 주변분포와 생존시간 분포를 참고한 합성 코호트
- 데이터 생성 시드당 환자 300명
- 최종 예측 입력: 나이와 치료군
- 관측 결과: 관측시간과 사건·검열 여부
- 사건과 검열 사례의 비율을 유지한 80%/10%/10% 학습·검증·테스트 분할

저장소에 포함된 실험자료는 모두 합성자료이며 실제 PBC 원자료는 포함하지 않았습니다. 자료 출처와 입력 형식은 [`data/README.md`](data/README.md)에 정리했습니다.

## Data Generation

합성자료 생성 과정은 두 단계로 정리했습니다.

1. `data/gen_synthetic.py`가 사용자가 별도로 지정한 PBC 자료에서 변수별 주변분포, Weibull 생존시간 분포와 검열률을 추정해 중간 합성 코호트를 만듭니다.
2. `data/generate_two_covariate_survival_data.py`가 나이와 치료군에 따른 Weibull 생존시간을 생성하고 데이터 시드별 최종 합성 코호트를 만듭니다.

두 번째 스크립트는 연구 당시 사용한 Weibull proportional-hazards simulation 코드를 공개 저장소에서 실행할 수 있도록 정리한 것입니다. 분포 추정에 사용한 원 PBC 파일은 저장소에 포함하지 않으며, 실행할 때 `--source-pbc`로 직접 지정해야 합니다. 생성되는 중간·재생성 자료는 Git에서 제외하고, 연구 당시 사용한 최종 합성자료만 재현 기록으로 보존합니다.

## Target Construction

검열된 관측시간을 그대로 정답으로 취급하지 않기 위해 다음 절차로 학습 목표를 구성했습니다.

1. 나이와 치료군을 공변량으로 Weibull AFT 모형을 적합합니다.
2. 로그 관측시간과 AFT 예측값의 잔차를 계산합니다.
3. 잔차의 생존분포를 Kaplan–Meier 방법으로 추정합니다.
4. 사건이 관측된 환자는 실제 로그시간을, 검열된 환자는 Buckley–James 방식에서 착안한 조건부 기대 잔여시간을 사용합니다.
5. 최종 실험에서는 로그 목표값을 다시 선형 시간 단위로 변환합니다.

이 목표값은 실제 미관측 생존시간을 복원한 참값이 아니라, 검열정보를 활용해 만든 분석용 추정 목표입니다.

## Model and Training

- Base model: `meta-llama/Llama-3.1-8B`
- Parameter-efficient fine-tuning: LoRA
- Quantization: 4-bit NF4 with double quantization
- Compute dtype: float16
- Epochs: 15
- Batch size: 4
- Model seed: 74
- LoRA rank: 16 or 32
- LoRA alpha: twice the rank
- Learning rate: `1e-4` or `5e-5`

모델에는 다음과 같은 형태의 프롬프트를 제공하고, 설명 없이 숫자 하나만 생성하도록 학습했습니다.

```text
The patient with id ... was enrolled in the PBC study.
At enrollment, the patient was ... years old and received ... .
Based on these features, predict the expected survival time (T).
Output only one non-negative real number. No extra text.
```

숫자로 해석할 수 없는 출력에는 재시도와 통계적 대체값을 적용하고, 원출력의 유효성과 최종 예측값을 별도로 기록했습니다.

## Experimental Design

최종 선형시간 실험은 세 개의 데이터 생성 시드에서 네 개의 LoRA 설정을 비교했습니다. 설정 선택 시 RMSE뿐 아니라 숫자 출력의 안정성도 함께 확인했습니다.

| Rank | Alpha | Learning rate | Mean test RMSE | SD | Mean raw invalid ratio | Mean greedy success rate |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | `5e-5` | **3.057** | 0.467 | **0.000** | **1.000** |
| 16 | 32 | `1e-4` | 3.698 | 1.330 | 0.011 | 0.809 |
| 32 | 64 | `5e-5` | 3.283 | 0.441 | **0.000** | 0.939 |
| 32 | 64 | `1e-4` | 2.848 | 0.324 | 0.667 | 0.333 |

전체 결과는 [`reports/final-linear-260120/`](reports/final-linear-260120/)에 보존했습니다.

## Results

대표 설정으로 rank 16, alpha 32, learning rate `5e-5`를 선택했습니다.

- 세 데이터 시드 평균 테스트 RMSE: **3.06 ± 0.47년**
- 원출력 invalid ratio: **0%**
- greedy generation 성공률: **100%**
- retry 및 통계적 fallback 사용률: **0%**

Rank 32와 learning rate `1e-4` 조합은 평균 RMSE가 더 낮았지만, 세 시드 중 두 시드에서 원출력 실패가 광범위하게 발생했습니다. 최종 유효값은 재시도 또는 대체 절차로 확보되었으므로, 이 설정을 가장 좋은 모델로 해석하지 않았습니다.

이 결과는 LLM이 전통적 생존모형보다 우수함을 입증하는 결과가 아니라, 검열 보정 목표에 대한 숫자 회귀와 출력 안정성을 탐색한 결과입니다.

## Limitations

- 합성자료만 사용했습니다.
- 최종 입력변수가 나이와 치료군으로 제한적입니다.
- 시드당 표본 수가 300명이고 최종 비교는 세 데이터 시드에 기반합니다.
- Cox, Weibull AFT, Random Survival Forest 등 동일 조건의 기준모형 비교가 없습니다.
- 생존곡선이 아니라 기대 생존시간 하나만 예측합니다.
- 외부자료 또는 실제 임상자료 검증이 없습니다.
- 모델 checkpoint와 원시 학습 로그는 포함하지 않았으며, 최종 결과표와 확인된 설정을 이용해 재현 스크립트를 구성했습니다.

## Repository Structure

```text
llm-survival-time-prediction/
├── data/                         # 합성 코호트와 파생 목표자료
│   ├── README.md                 # 자료 출처·입력 형식·생성 과정
│   ├── gen_synthetic.py          # PBC 분포 기반 중간 합성자료 생성
│   └── generate_two_covariate_survival_data.py
├── regression/
│   ├── make_target.py            # 검열 보정 목표 생성
│   ├── data_utils_share.py       # 프롬프트·출력 파싱·fallback
│   ├── llama_finetuner_logit_cp_share.py
│   ├── main_logit_cp_share.py    # 학습과 평가 진입점
│   ├── run_final_linear_grid.sh  # 최종 선형시간 설정 재현
│   └── summarize_grid_results.py # 결과 수집과 요약
├── reports/
│   └── final-linear-260120/      # 최종 compact result tables
└── environment_final.yaml
```

## Reproduction

CUDA를 사용할 수 있는 환경과 Hugging Face 접근 권한이 필요합니다.

먼저 실행환경을 구성합니다.

```bash
conda env create -f environment_final.yaml
conda activate survlift_final
```

합성자료 생성 과정을 처음부터 실행하려면 원 PBC CSV 경로를 지정합니다.

```bash
python data/gen_synthetic.py \
  --source-pbc /path/to/pbc_with_13_covariates.csv

python data/generate_two_covariate_survival_data.py \
  --source-pbc /path/to/pbc_with_13_covariates.csv \
  --base-synthetic data/synthetic_pbc_data_N300.csv
```

기본적으로 두 번째 스크립트의 출력은 `data/generated/`에 저장되며 Git에는 포함되지 않습니다. 저장소에 보존된 연구 당시 합성자료를 이용해 학습 목표와 실험을 재현하려면 다음 명령을 사용합니다.

```bash
python regression/make_target.py --target-scale linear

export HF_TOKEN="your_huggingface_token"
bash regression/run_final_linear_grid.sh
```

실험 날짜 폴더의 결과를 다시 모으려면 다음 명령을 사용합니다.

```bash
python regression/summarize_grid_results.py \
  --collect \
  --results-root regression/results \
  --experiment YYMMDD

python regression/summarize_grid_results.py \
  --csv regression/results/YYMMDD/final_summary_results.csv
```

## My Contributions

- PBC 특성을 참고한 합성 생존자료와 검열 보정 목표 구성
- 생존시간 숫자 생성을 위한 임상 텍스트 프롬프트 설계
- 4-bit Llama 3.1 8B의 LoRA 미세조정 파이프라인 구현
- 데이터 시드와 LoRA 설정에 따른 반복 실험 수행
- RMSE와 생성 유효성을 함께 사용한 결과 검토
