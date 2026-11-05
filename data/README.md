# data/

이 디렉터리의 내용은 git에서 제외됩니다. 추적하는 파일은 이 README와 `data/metadata/videos.csv` 두 개뿐입니다. 영상에 사람이 찍혀 있으므로 외부 공유는 사전 승인을 받습니다.

- 원본 영상은 수정하지 않습니다. 파생 데이터는 별도로 저장합니다.
- 영상 선택과 split의 기준은 `data/metadata/videos.csv`입니다. 그룹, 사람 ID, 수준, 반복 정보는 폴더 이동이 아니라 manifest 열로 관리합니다.
- `data/metadata/videos.csv`는 git이 추적하므로 팀원이 같은 파일에 직접 입력할 수 있습니다. `scripts/import_archive_data.py`는 이 파일을 덮어쓰지 않습니다.
- 구조와 받는 방법은 데이터 이동 작업(후속 Task)에서 이 문서에 채웁니다. 현재는 미정입니다.

## `near_side` 열 (이슈 #7)

- 영상에서 카메라에 가까운 쪽이 수행자의 왼쪽인지 오른쪽인지를 영상 단위로 기록합니다. 값은 `left` 또는 `right`입니다.
- 입력 담당은 도메인 담당(`scv0413`)이고, 판단은 영상을 보고 합니다. `unknown`은 쓰지 않습니다.
- 입력 전에는 빈 값입니다. `load_manifest`는 빈 값을 허용하지만 `left`, `right`가 아닌 값(대소문자 포함)은 오류로 처리합니다.
- 가까운 쪽이 필요한 단계는 `require_near_side`를 호출하며, 빈 값이 남아 있으면 해당 영상 id를 나열하고 멈춥니다. 사람 단위 분할은 이 값을 쓰지 않습니다.
- 이 파일의 `visible_body_side` 열은 아카이브에서 가져온 값(left 6개, unknown 48개)이며 `near_side`와 다른 열입니다. `near_side`의 근거로 쓰지 않습니다.
