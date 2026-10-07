# Final Linear-Time Experiment Results

2026년 1월 20일 완료된 선형 생존시간 예측 실험의 요약 결과입니다.

## Provenance

- Original experiment date: 2026-01-20
- Model: `meta-llama/Llama-3.1-8B`
- Data seeds: 1, 2, 3
- Model seed: 74
- Target scale: linear time in years
- Copied into this report: 2026-10-07

핵심 학습·평가 코드와 결과표를 보존했지만, 당시의 모델 checkpoint와 원시 학습 로그를 포함한 완전한 실행 스냅샷은 아닙니다. `regression/run_final_linear_grid.sh`는 결과표와 확인된 실험 설정을 명시적으로 재구성합니다.

## Files

- `final_summary_results.csv`: 설정·데이터 시드별 개별 결과
- `final_summary_results_stats.csv`: 설정별 평균, 표준편차, 시드 수
- `final_summary_results_stats.txt`: 사람이 읽기 쉬운 전체 요약표

모델 가중치, tokenizer, 원시 로그 및 중간 checkpoint는 포함하지 않았습니다.

## Representative Configuration

안정적인 숫자 생성과 RMSE를 함께 고려해 다음 설정을 대표 결과로 사용합니다.

| Setting | Value |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | `5e-5` |
| Mean test RMSE | 3.057 years |
| RMSE SD | 0.467 years |
| Raw invalid-answer ratio | 0.000 |
| Greedy success rate | 1.000 |
| Number of data seeds | 3 |

## Interpretation

Rank 32, learning rate `1e-4` 설정은 평균 RMSE 2.848로 가장 낮았지만 raw invalid-answer ratio가 높았습니다. 재시도와 통계적 대체 절차 이후에는 유효한 숫자를 얻었지만, 모델 자체의 안정적인 생성 성능으로 해석하지 않았습니다.

이 결과는 기준 생존모형과의 우월성 비교가 아니라, 검열 보정 목표에 대한 LoRA 회귀 가능성과 숫자 출력 안정성을 확인한 탐색 결과입니다.
