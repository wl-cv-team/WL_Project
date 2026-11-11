# data/

영상에 사람 얼굴이 나오므로 **얼굴이 담긴 이미지는 어떤 형태로도(흐리게 처리한 것 포함) 저장소에 올리지 않습니다.** 영상과 영상에서 뽑은 이미지(jpg, png 등)가 대상입니다. 영상은 팀 Google Drive "영상 공유" 폴더로 공유하고, 외부 공유는 사전 승인을 받습니다. 이 저장소는 공개 상태입니다.

라벨이 맞는지는 로컬 영상에서 프레임을 읽어 화면으로만 확인하고, 이미지를 파일로 저장해 올리지 않습니다.

이미지가 올라갈 수 있는 다른 경로도 같은 원칙으로 막습니다.
- 노트북 출력(`.ipynb`에 저장된 그림)
- PR과 이슈 댓글의 스크린샷, 시각화 첨부. 첨부한 이미지는 git이 아니라 GitHub 서버에 올라가서 `.gitignore`로 막을 수 없습니다. **얼굴이 보이는 스크린샷이나 노트북 화면 캡처를 이슈와 PR에 첨부하지 않습니다.**
- `docs/`와 README의 예시 이미지(필요하면 얼굴이 없는 도식이나 좌표 그림을 쓴다)
- `artifacts/`의 출력물. git에서 제외되어 있습니다. 이미지를 파일로 만드는 코드는 두지 않고(이전의 컨택트 시트 스크립트는 삭제), 확인은 노트북 화면에서만 합니다.

`.gitignore`는 영상과 이미지 확장자(mp4, mov, avi, mkv, jpg, jpeg, png, webp, gif, bmp, tif, tiff, heic)를 대문자 확장자까지 `data/` 아래 어디서든 제외합니다.

git이 추적하는 것은 이 README와 아래 세 폴더의 표와 텍스트 파일입니다. 목록에 없는 폴더와 파일은 추적하지 않습니다.

| 폴더 | 내용 |
|---|---|
| `metadata/` | `videos.csv`(영상 목록, `near_side` 입력) |
| `annotations/` | 라벨 표와 텍스트 라벨 |
| `splits/` | 사람 단위 분할 표 |

`raw/`, `plate_dataset/`, `plate_dataset_v2/`는 추적하지 않습니다. `annotations/`와 `splits/`에 있는 `phase_events.csv`, `pilot_frame_ground_truth.csv`, `u11_task3_plate_identity_labels/`, `evaluation_videos.csv`는 아카이브에서 가져온 파일입니다. 기존 라벨은 사용하지 않기로 했고(`docs/decisions/0002`), 새 라벨로 대체될 때까지 보관만 합니다.

- 원본 영상은 수정하지 않습니다. 파생 데이터는 별도로 저장합니다.
- 영상 선택과 split의 기준은 `data/metadata/videos.csv`입니다. 그룹, 사람 ID, 수준, 반복 정보는 폴더 이동이 아니라 manifest 열로 관리합니다.
- `data/metadata/videos.csv`는 git이 추적하므로 팀원이 같은 파일에 직접 입력할 수 있습니다. `scripts/import_archive_data.py`는 이 파일을 덮어쓰지 않습니다.
- 구조와 받는 방법은 데이터 이동 작업(후속 Task)에서 이 문서에 채웁니다. 현재는 미정입니다.

## 점 라벨 `annotations/point_labels.csv` (이슈 #8, #12)

`tutorials/label_points.ipynb`로 도메인 담당이 영상마다 인상 구간 경계 프레임을 직접 찍은 결과입니다. 열은 `video_id, event, frame, pass`이고, event는 `ready, transition, drive, catch, finish`, pass는 1 또는 2입니다. `frame`은 디코딩 순서의 0부터 시작하는 번호입니다(#10). `(video_id, event, pass)`가 모두 같을 때만 새 값이 옛 값을 대체하고, pass 1과 pass 2는 같은 프레임이어도 둘 다 보존합니다. 같은 영상에 pass 2 기록이 하나라도 있으면 그 영상의 pass 1은 추가도 수정도 할 수 없습니다. `completion`이 `h`인 영상은 `ready`, `transition`, `drive`만 필수이고 `catch`, `finish`는 "ignore"입니다. ignore는 파일에 행으로 남기지 않고 `completion` 값에서 도출하므로 기록이 없는 "누락"과 구별되며, 이 두 event를 기록하려 하면 오류입니다.

## `near_side` 열 (이슈 #7)

- 영상에서 카메라에 가까운 쪽이 수행자의 왼쪽인지 오른쪽인지를 영상 단위로 기록합니다. 값은 `left` 또는 `right`입니다.
- 입력 담당은 도메인 담당(`scv0413`)이고, 판단은 영상을 보고 합니다. `unknown`은 쓰지 않습니다.
- 입력 전에는 빈 값입니다. `load_manifest`는 빈 값을 허용하지만 `left`, `right`가 아닌 값(대소문자 포함)은 오류로 처리합니다.
- 가까운 쪽이 필요한 단계는 `require_near_side`를 호출하며, 빈 값이 남아 있으면 해당 영상 id를 나열하고 멈춥니다. 사람 단위 분할은 이 값을 쓰지 않습니다.
- `videos.csv`의 `fps` 열은 프레임 수를 컨테이너 재생 길이로 나눈 평균값으로 추정되며 실제 프레임 간격과 다릅니다(예: 27.26인데 실제 간격은 33.33ms). **시간 계산에 쓰지 않습니다.** 시간은 디코딩한 프레임의 타임스탬프로 구합니다(#10).
- 이 파일의 `visible_body_side` 열은 아카이브에서 가져온 값(left 6개, unknown 48개)이며 `near_side`와 다른 열입니다. `near_side`의 근거로 쓰지 않습니다.
