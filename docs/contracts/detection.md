# 탐지 결과 형식

구현: `src/wl_snatch/contracts.py`. 탐지 단계가 이 형식으로 결과를 쓰고, 평가와 다음 단계(바 경로)가 이 형식을 입력으로 읽는다.

## CSV 형식

열 순서: `video_id, frame_index, x1, y1, x2, y2, confidence, label, na_reason`

- 한 행은 박스 하나다.
- 박스가 있는 프레임은 박스 수만큼 행을 쓴다. 이 행들의 `na_reason`은 비어 있다.
- 박스가 없는 프레임도 프레임이 존재했다는 사실을 남기기 위해 한 행을 쓴다. 이 행의 좌표, `confidence`, `label`은 비어 있다.
- 읽을 때 프레임의 순서와 `video_id`를 그대로 유지한다. 프레임은 `(video_id, frame_index)`로 식별한다.

## 좌표 규칙

- 단위는 픽셀이다.
- `x1, y1`은 박스 왼쪽 위 모서리, `x2, y2`는 오른쪽 아래 모서리다.
- 원점은 이미지의 왼쪽 위이고, x는 오른쪽으로, y는 아래로 증가한다.

## N/A 규칙

- 측정할 수 없는 프레임(예: 읽을 수 없는 프레임)은 `na_reason`에 사유를 적는다. 좌표 열은 비운다.
- N/A는 탐지 0개와 다르다.
  - 탐지 0개: 프레임을 처리했고 박스가 없다. 좌표와 `na_reason`이 모두 비어 있다.
  - N/A: 프레임을 처리하지 못했다. 좌표는 비어 있고 `na_reason`이 채워져 있다.
- N/A 프레임은 박스를 가지지 않는다. `na_reason`이 있는데 박스가 있으면 `FrameDetections` 생성 시 `ValueError`가 발생한다.
- 평가에서 N/A 프레임은 평균에서 제외하고 개수를 따로 보고한다.

## 읽을 때 검사하는 규칙

- 같은 `(video_id, frame_index)`의 행은 연속해야 한다. 떨어진 위치에 같은 키가 다시 나오면 `ValueError`다.
- 같은 프레임의 행들은 `na_reason`이 같아야 한다. 박스 없는 행은 그 프레임의 유일한 행이어야 한다.
- 한 행에서 좌표, `confidence`, `label`은 모두 채우거나 모두 비워야 한다. `na_reason`이 있으면 좌표를 채울 수 없다.
- 숫자가 아닌 좌표, 필수 열 누락도 `ValueError`다. 메시지에 파일 경로와 행 번호(헤더가 1행)가 들어간다.
- `na_reason`은 `None` 또는 비어 있지 않은 문자열이다. 빈 문자열은 허용하지 않는다.
- 박스는 NaN을 가질 수 없고 `x1 <= x2`, `y1 <= y2`여야 한다. 음수 좌표는 허용한다.

## N/A 사유 값

| 값 | 만드는 곳 | 뜻 |
|---|---|---|
| `video_unreadable` | 탐지기(`detect_frames`, `detect_video`) | 영상을 열 수 없다. `detect_frames`는 요청한 모든 프레임(음수 포함)에, `detect_video`는 프레임 0 하나에 쓴다 |
| `frame_out_of_range` | 탐지기(`detect_frames`) | 요청한 프레임 번호가 음수이거나 디코딩한 프레임 수 이상이다 |
| `prediction_missing` | 평가기(`evaluate_detections`) | 정답 프레임에 대응하는 예측 항목이 없다. 탐지 CSV에는 쓰이지 않고 `DetectionMetrics.na_reasons`에만 나타난다 |

같은 실패를 다른 이름으로 쓰지 않는다. 사유 값을 더하면 이 표와 `tests/`를 함께 고친다.

## 정답 형식 `GroundTruthFrame`

구현: `src/wl_snatch/contracts.py`(`eval_detection`에서도 가져올 수 있다). 정답은 탐지 결과 CSV가 아니라 라벨 파일(`data/annotations/plate_boxes/`)과 분할 표에서 `ground_truth.load_ground_truth`가 만든다.

- `GroundTruthFrame(video_id, frame_index, boxes)`: 프레임 하나의 정답. `boxes`는 `(x1, y1, x2, y2)` 픽셀 좌표(위의 좌표 규칙과 같다) 튜플이다.
- `boxes`가 비어 있으면 가까운 쪽 원판이 없다고 확인한 프레임이다. 평가에서 이 프레임의 예측은 모두 오탐이다.
- `frame_index`는 `FrameDetections`와 같은 뜻이다(디코딩 순서, 0부터). `(video_id, frame_index)`로 식별한다.

## 다음 단계

바 경로 단계는 이 CSV(`read_detections_csv`가 반환하는 `list[FrameDetections]`)를 입력으로 받는다.
