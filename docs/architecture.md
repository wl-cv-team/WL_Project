# 구조

탐지 베이스라인을 기준으로 한 코드의 층과 처리 순서를 적는다. 구현 순서와 Task 번호는 `docs/superpowers/plans/2026-11-05-foundation-and-detection.md`를 따른다. 표의 "상태"는 이 문서를 쓴 시점의 코드를 기준으로 한다.

- **있음:** 코드와 테스트가 있다.
- **A단계:** 이 문서와 같은 변경에서 구현했다(Task 6, 6a, 7, 7b, 7c).
- **예정(B단계):** 아직 구현하지 않았다. 이슈 #2의 학습 구성과 이미지 정책 답변 뒤에 한다(Task 6b, 8).

## 처리 순서

```
라벨 파일 plate_boxes/*.txt ─┐
영상 목록 videos.csv ────────┼─▶ 정답 만들기 ─▶ GroundTruthFrame 목록 ─┬─▶ 데이터셋 내보내기 ─▶ 학습용 폴더(로컬, git 제외)
분할 표 person_split.csv ────┘   (Task 6a)                              │    (Task 6b, 예정)         영상 파일이 입력
                                                                         │
영상 파일 ─▶ 프레임을 메모리에서 읽기 ─▶ 탐지기 ─▶ FrameDetections ─▶ 평가기 ─▶ DetectionMetrics ─▶ 비교 ─▶ CSV와 결정 기록
              (frames.read_frames)       (Task 7)   (contracts.py)    (Task 6)                      (Task 8, 예정)
                                                  └─────────── 정답으로 평가하기: evaluation_run (Task 7c) ───────────┘
```

- 정답 만들기의 입력은 라벨 파일, `videos.csv`(영상 크기와 프레임 수), 분할 표다. 영상 파일은 읽지 않는다. 영상 파일은 데이터셋 내보내기(예정)와 탐지기가 읽는다.
- 단계 사이는 `contracts.py`의 데이터 형식(`BoxDetection`, `FrameDetections`, `GroundTruthFrame`)으로만 주고받는다. 다음 단계(바 경로 등)는 `docs/contracts/detection.md`의 형식을 입력으로 받는다.
- 모델은 "프레임 하나를 받아 `(x1, y1, x2, y2, confidence)` 목록을 돌려주는 호출 가능 객체"(`ModelFn`)로 주입한다. 모델을 바꿔도 평가, 비교, 정답 만들기는 바뀌지 않는다.
- `videos.csv`의 `width`, `height`, `frame_count`는 디코딩한 값과 같아야 한다. `video_timing.csv`와 54편 모두 일치함을 확인했지만 코드가 다시 검사하지는 않는다.

## 층

| 층 | 모듈 | 책임 | 상태 |
|---|---|---|---|
| 약속 | `contracts.py` | 단계 사이 데이터 형식, `N/A`와 "탐지 0개"의 구분, 정답 형식 | 있음 |
| 영상 읽기 | `frames.py` | 디코딩 순서의 프레임 번호, 프레임 읽기(한 번 순차 디코딩), 시간 정보. 이미지를 파일로 쓰지 않는다 | 있음, `read_frames`와 `check_readable`은 A단계 |
| 영상 목록과 분할 | `manifest.py`, `split.py` | `videos.csv` 검증, 사람 단위 분할과 누수 검사 | 있음 |
| 라벨 | `plate_boxes.py`, `labeling.py` | 원판 박스와 단계 경계 점 라벨의 저장과 읽기 | 있음 |
| 라벨 점검과 보기 | `label_audit.py`, `label_viewer.py` | 아카이브 라벨 점검, 박스를 메모리에서 그리기 | 있음 |
| 데이터 가져오기와 시간 정보 | `data_import.py`, `video_timing.py` | 아카이브 가져오기, 영상 시간 간격 표 | 있음 |
| 정답 만들기 | `ground_truth.py` | 라벨, 영상 목록, 분할에서 `GroundTruthFrame` 목록을 만든다 | A단계(Task 6a) |
| 평가 | `eval_detection.py` | IoU 매칭, 정밀도, 재현율, 프레임당 오탐, `N/A` 집계 | A단계(Task 6) |
| 탐지기 | `config.py`, `detector.py` | 설정으로 모델과 장치를 고르고, 프레임에서 탐지한다. 단일 클래스 `plate` 가중치만 쓴다 | A단계(Task 7) |
| 정답으로 평가하기 | `evaluation_run.py` | 정답을 영상별로 묶어 탐지하고 평가한다 | A단계(Task 7c) |
| COCO 사전학습 그대로 실행 | `coco_baseline.py` | 원래 COCO 클래스 이름을 유지해 실행하고 클래스별 개수와 추론 시간을 기록한다. 원판 평가와 연결하지 않는다 | 예정(Task 8a) |
| 데이터셋 내보내기 | `dataset.py` | 학습용 폴더와 `dataset.yaml`을 `data/derived/` 아래에 만든다 | 예정(Task 6b, fine-tuning을 시작할 때) |
| 학습 | `train.py` | 사전학습 가중치에서 시작해 `plate` 한 클래스로 학습하고 가중치 경로를 돌려준다 | 예정(Task 8) |
| 비교 | `compare.py` | 한 번에 한 축만 바꾼 비교를 강제하고 CSV로 기록한다 | 예정(Task 8) |
| 실행기 | `scripts/*.py` | 위 모듈을 호출만 한다. 로직을 두지 않는다 | 있음, 탐지 실행기는 A단계 |
| 노트북 | `tutorials/*.ipynb` | `src/`를 불러 쓰기만 한다. 정의, 긴 셀, 저장된 출력을 두지 않는다 | 있음 |

## 정책이 지켜지는 곳

| 규칙 | 지켜지는 곳 | 상태 |
|---|---|---|
| 얼굴이 담긴 이미지는 저장소에 올리지 않는다 | `.gitignore`(`data/` 아래 이미지 확장자와 `runs/` 제외), PR 템플릿, 이미지를 파일로 쓰지 않는 것을 확인하는 모듈별 테스트 | 있음 |
| 평가와 탐지는 이미지 파일 없이 메모리에서 한다 | `frames.read_frames`, `detect_frames` | 있음 |
| 학습용 이미지는 로컬 git 제외 폴더에만 둔다(2026-11-18 합의) | `dataset.assert_ignored_output` | 예정(Task 6b) |
| 같은 사람이 학습과 검증 양쪽에 없다 | 분할 표를 만들 때 `split.assert_no_person_leak`. `ground_truth.load_ground_truth`는 분할 표의 `role`을 그대로 따르며 `person_id`를 다시 검사하지 않는다 | 있음 |
| 측정하지 못한 값은 채우지 않고 `N/A`와 사유를 남긴다 | `contracts.py`, `detector.py`, `eval_detection.py`(사유 값은 `docs/contracts/detection.md`) | 있음. 비교 CSV는 예정(Task 8) |
| 한 번에 한 축만 바꾸고 같은 프레임에서 비교한다 | `compare.assert_single_axis` | 예정(Task 8) |
| 노트북에 로직이 없다 | `tests/test_notebook_hygiene.py` | 있음 |
| 모델과 라이브러리는 설정으로 교체하고 코드에 고정하지 않는다 | `DetectorConfig`, `ModelFn` 주입 | 있음 |
| 사전학습 가중치로 원판을 탐지했다고 쓰지 않는다 | `detector.check_plate_names`, `load_ultralytics_model`(`weights_path` 필수) | A단계 |
| 층 사이 약속이 어긋나지 않는다 | `tests/test_pipeline.py`(정답 만들기 → 탐지 → 평가) | A단계 |

## 의존성

`torch`, `torchvision`, `ultralytics`는 기본 의존성이다. 탐지기 모듈은 Ultralytics를 함수 안에서 가져오므로(`load_ultralytics_model`), 라벨링과 평가만 쓰는 코드는 모델을 불러오지 않는다. 선택 그룹으로 옮기는 일은 팀이 정할 때 별도 변경으로 한다.

## 확장하는 곳

- **새 모델 구조:** `config.WEIGHT_NAMES`에 사전학습 가중치 이름을 더한다(학습의 시작점). 학습한 가중치는 `DetectorConfig.weights_path`로 지정한다. 평가, 비교, 정답 만들기는 그대로다.
- **새 평가 지표:** `eval_detection.py`에 더하고 `DetectionMetrics`와 비교 CSV의 열을 함께 바꾼다.
- **다음 단계(바 경로 등):** `FrameDetections` CSV를 입력으로 받는다. 형식을 바꾸면 `docs/contracts/detection.md`와 이 형식을 쓰는 단계의 코드와 테스트를 같이 바꾼다.
- **다른 분할이나 라벨 묶음:** `person_split.csv`와 `plate_boxes/`가 입력이므로 파일만 바꾼다.

## 정해진 것과 아직 정해지지 않은 것
정해진 것(`docs/decisions/0003-baseline-setup.md`): 학습 45장(원판 있음 39·없음 6), 검증 39장, 빈 라벨은 학습용, 학습 이미지는 로컬 git 제외 폴더에 생성 허용, 첫 실행은 fine-tuning 없는 COCO 사전학습 그대로 실행.

아직 정해지지 않은 것:
- fine-tuning 설정(backbone 동결, epoch, 학습률, 배치, seed). COCO 실행 결과를 본 뒤 제안한다
- 원판 없음 독립 검증용 별도 인원의 라벨
- 두 번째 라벨링(pass 2)이 합쳐진 뒤 어느 pass를 정답으로 쓸지
- TCN 적용 위치(이 구조의 바깥, `docs/decisions/0002`)
