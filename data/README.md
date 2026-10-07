# Data

이 저장소에는 연구에 사용한 합성 생존자료만 포함하며, 실제 PBC 환자자료는 재배포하지 않습니다.

## Source Dataset

합성자료의 변수분포, 생존시간 분포와 검열률은 R `survival` 패키지의 Mayo Clinic Primary Biliary Cholangitis 자료를 참고했습니다.

- Official documentation: https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbc.html
- R usage: `data(pbc, package = "survival")`

원자료에서 생성 스크립트로 전달하는 CSV는 다음 열을 사용합니다.

```text
time, status, trt, age, sex, ascites, hepato, spiders,
edema, stage, bili, albumin, alk.phos, ast, protime
```

- `time`: 추적기간(일)
- `status`: `0`은 검열, `1`은 사건으로 정리한 이진 지표
- `trt`: `1` 또는 `2`로 표현한 치료군
- 나머지 열: PBC 문서에 정의된 기초 임상변수

원 PBC 자료의 `status`는 `0/1/2`로 제공되므로, 실제 실행 전 연구 정의에 맞는 사건을 선택해 이진 지표로 변환해야 합니다. 이 저장소는 특정 변환을 유일한 정답으로 제시하거나 변환된 실제 환자 CSV를 배포하지 않습니다.

## Included Synthetic Cohorts

`synthetic_pbc_data_N300_given_2_covariate_seed1.csv`부터 `seed10.csv`까지는 연구 당시 생성한 합성 코호트입니다.

- 데이터 시드별 300명
- 최종 공변량: 나이와 치료군
- `true_T`: 생성된 잠재 생존시간
- `C`: 생성된 검열시간
- `time`: `min(true_T, C)`
- `status`: 사건 관측 여부

합성자료는 실제 환자 기록이 아니며 개인 식별정보를 포함하지 않습니다.

## Generation Pipeline

1. `gen_synthetic.py`: 원 PBC 자료의 주변분포, Weibull 생존시간 분포와 검열률을 참고해 중간 합성자료 생성
2. `generate_two_covariate_survival_data.py`: 나이와 치료군에 따른 데이터 시드별 합성 생존시간 생성
3. `regression/make_target.py`: 검열정보를 이용해 LLM 학습용 `target_time` 생성

중간자료, 재생성 자료, 파생 target과 추정 계수는 `.gitignore`로 제외합니다.
