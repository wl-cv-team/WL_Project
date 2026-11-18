# 구조

탐지 베이스라인을 기준으로 한 코드의 층과 데이터가 흐르는 방향을 적는다. 구현 순서와 Task 번호는 `docs/superpowers/plans/2026-11-05-foundation-and-detection.md`를 따른다.

## 데이터가 흐르는 방향

```
영상(로컬) ──────┐
원판 박스 라벨 ──┼─▶ 정답 만들기 ─▶ GroundTruthFrame 목록 ─┬─▶ 데이터셋 내보내기 ─▶ 학습용 폴더(로컬, git 제외)
person_split.csv ┘   (Task 6a)                              │      (Task 6b)
                                                            │
영상 프레임(메모리) ─▶ 탐지기 ─▶ FrameDetections ─▶ 평가기 ─▶ DetectionMetrics ─▶ 비교 ─▶ CSV와 결정 기록
                      (Task 7)   (contracts.py)     (Task 6)                      (Task 8)
```

- 단계 사이는 `contracts.py`의 데이터 형식(`BoxDetection`, `FrameDetections`)과 정답 형식(`GroundTruthFrame`)으로만 주고받는다. 다음 단계(바 경로 등)는 `docs/contracts/detection.md`의 형식을 입력으로 받는다.
- 모델은 "프레임 하나를 받아 `(x1, y1, x2, y2, confidence)` 목록을 돌려주는 호출 가능 객체"(`ModelFn`)로 주입한다. 모델을 바꿔도 평가, 비교, 정답 만들기는 바뀌지 않는다.

## 층

| 층 | 모듈 | 책임 |
|---|---|---|
| 약속 | `contracts.py` | 단계 사이 데이터 형식, `N/A`와 "탐지 0개"의 구분 |
| 영상 읽기 | `frames.py` | 디코딩 순서의 프레임 번호, 프레임 읽기, 시간 정보. 이미지를 파일로 쓰지 않는다 |
| 영상 목록과 분할 | `manifest.py`, `split.py` | `videos.csv` 검증, 사람 단위 분할 |
| 라벨 | `plate_boxes.py`, `labeling.py` | 원판 박스와 단계 경계 점 라벨의 저장과 읽기 |
| 정답 만들기 | `ground_truth.py` (Task 6a) | 영상, 박스 라벨, 분할에서 `GroundTruthFrame` 목록을 만든다 |
| 평가 | `eval_detection.py` (Task 6) | IoU 매칭, 정밀도, 재현율, 프레임당 오탐, `N/A` 집계 |
| 탐지기 | `config.py`, `detector.py` (Task 7) | 설정으로 모델과 장치를 고르고, 프레임에서 탐지한다 |
| 데이터셋 내보내기 | `dataset.py` (Task 6b) | 학습용 폴더와 `dataset.yaml`을 만든다(이미지 정책 결정 뒤) |
| 비교 | `compare.py` (Task 8) | 한 번에 한 축만 바꾼 비교를 강제하고 CSV로 기록한다 |
| 실행기 | `scripts/*.py` | 위 모듈을 호출만 한다. 로직을 두지 않는다 |
| 노트북 | `tutorials/*.ipynb` | `src/`를 불러 쓰기만 한다. 정의, 긴 셀, 저장된 출력을 두지 않는다 |

## 정책이 지켜지는 곳

| 규칙 | 지켜지는 곳 |
|---|---|
| 얼굴이 담긴 이미지는 저장소에 올리지 않는다 | `.gitignore`(`data/` 아래 이미지 확장자 제외), PR 템플릿, 이미지를 파일로 쓰지 않는 모듈별 테스트 |
| 평가와 탐지는 이미지 파일 없이 메모리에서 한다 | `frames.read_frames`, `detect_frames` |
| 학습용 이미지는 로컬 git 제외 폴더에만 둔다(정책 결정 뒤) | `dataset.assert_ignored_output` |
| 같은 사람이 학습과 검증 양쪽에 없다 | `split.py`, `ground_truth.load_ground_truth`(역할별 반환) |
| 측정하지 못한 값은 채우지 않고 `N/A`와 사유를 남긴다 | `contracts.py`, `eval_detection.py`, 비교 CSV |
| 한 번에 한 축만 바꾼다 | `compare.assert_single_axis` |
| 노트북에 로직이 없다 | `tests/test_notebook_hygiene.py` |
| 모델과 라이브러리는 설정으로 교체하고 코드에 고정하지 않는다 | `DetectorConfig`, `ModelFn` 주입 |

## 의존성

`torch`, `torchvision`, `ultralytics`는 기본 의존성이다. 탐지기 모듈은 Ultralytics를 함수 안에서 가져오므로(`load_ultralytics_model`), 라벨링과 평가만 쓰는 코드는 모델을 불러오지 않는다. 선택 그룹으로 옮기는 일은 팀이 정할 때 별도 변경으로 한다.

## 확장하는 곳

- **새 모델:** `config.WEIGHT_NAMES`에 이름을 더한다. 평가, 비교, 정답 만들기는 그대로다.
- **새 평가 지표:** `eval_detection.py`에 더하고 `DetectionMetrics`와 비교 CSV의 열을 함께 바꾼다.
- **다음 단계(바 경로 등):** `FrameDetections` CSV를 입력으로 받는다. 형식을 바꾸면 `docs/contracts/detection.md`와 이 형식을 쓰는 단계의 코드와 테스트를 같이 바꾼다.
- **다른 분할이나 라벨 묶음:** `person_split.csv`와 `plate_boxes/`가 입력이므로 파일만 바꾼다.

## 아직 정해지지 않은 것
- 학습용 이미지를 로컬에 만드는 방식(이슈 #2 질문 3)
- 베이스라인 학습 구성(학습 6편과 검증 6편, 음성 샘플의 용도)
- TCN 적용 위치(이 구조의 바깥, `docs/decisions/0002`)
