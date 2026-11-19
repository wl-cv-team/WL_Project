# 기초 구축과 탐지 기준선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 협업 가능한 저장소 뼈대와 데이터 기반을 만들고, 원판 탐지 기준선(YOLO26 → YOLO11 → YOLOv8 비교)을 같은 평가 조건으로 확보한다.

**Architecture:** 로직은 모두 `src/wl_snatch/`의 `.py` 모듈에 둔다. `scripts/`는 모듈을 호출만 하는 실행기이고, `tutorials/`의 노트북은 모듈을 불러다 쓰기만 한다. 단계 사이는 `contracts.py`의 데이터 형식으로만 주고받는다.

**Tech Stack:** uv, Python(버전은 Task 1에서 확인), Ultralytics, OpenCV, NumPy, pandas, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-project-foundation-design.md`

## Global Constraints

- 노트북에는 TDD 코드와 로직을 두지 않는다. 간단한 조건문만 허용한다. (spec 6절)
- 사람 단위로 학습/검증을 분할한다. 같은 `person_id`를 양쪽에 넣지 않는다. (spec 6절)
- 측정이 불가능하면 값을 채우지 않고 `N/A`와 사유를 남긴다. (spec 8절)
- `data/`는 git 제외. 원본 영상은 수정하지 않는다. (spec 5·8절)
- 좋은/나쁜 동작을 단정하는 문장과 비유적 표현을 문서, 코드, 커밋 메시지, 출력 라벨에 쓰지 않는다. (spec 4절)
- 모델 종류는 코드에 고정하지 않고 설정으로 교체한다. (spec 3절)
- 모델과 라이브러리 버전은 기억이 아니라 2026년 11월 기준 공식 문서로 확인해서 적는다. (spec 9-1절)
- 한 번에 한 축만 바꾸고 seed와 조건을 기록한다. (spec 9-2절)
- 기본 환경은 MacBook M2 Air 8GB다. (spec 5절)
- `main` 직접 push 금지. `feature/<주제>` 브랜치와 PR 상호 리뷰. (spec 5절)
- TCN 구현, 포즈 추정, 바 경로, 구간 분할은 이 계획에 넣지 않는다. (spec 9-1절)
- 학습용 프레임 이미지는 git 제외 로컬 폴더(`data/derived/`)에만 만들 수 있다(이슈 #2, 2026-11-18 합의). 같은 프레임을 만드는 추출 코드는 저장소에서 공유하고, 생성한 이미지와 영상은 git, 노트북 출력, PR·이슈 첨부에 올리지 않는다. 평가와 탐지는 프레임을 메모리에서 읽어 처리한다.

## Review Focus

1. **경로 탈출이 있는 zip.** `../` 또는 절대경로 항목이 든 압축은 복사하지 않고 `ValueError`로 거부한다. (Task 3)
2. **영상이 1개뿐인 사람, 사람이 1명뿐인 그룹.** 분할은 결정적이고 누수가 없으며, 한쪽이 비는 경우를 보고한다. (Task 4)
3. **라벨이 비었거나 범위를 벗어났거나 이미지와 짝이 없는 경우.** 죽지 않고 보고서에 사유와 함께 담는다. (Task 5)
4. **원판이 없는 영상(`has_plate=no`)에서 박스가 나온 경우, 읽을 수 없는 영상.** 앞의 경우는 오탐으로 센다. 뒤의 경우는 `N/A`와 사유로 남기고 평균에서 제외하며 개수를 보고한다. (Task 6)
5. **요청한 장치(mps/cuda)를 쓸 수 없는 경우.** 조용히 cpu로 바꾸지 않고, 바뀐 사실과 이유를 기록한다. (Task 7)

---

## Checkpoint (Task 5 직후)

Task 5의 라벨 점검 보고서를 팀이 보고 `docs/decisions/0002-label-reuse-and-tcn-gate.md`에 다음을 기록한다. 이 문서가 없으면 Task 6 이후를 진행하지 않는다.
1. 아카이브 라벨 재사용 여부, 재라벨링 필요 시 담당자와 규모.
2. TCN 적용 위치 승인 절차와 "시도했으나 효과 없음" 결과의 허용 여부. (이 계획은 TCN을 구현하지 않으므로 기록만 한다.)

**2026-11-06 결과:** `0002`에 기록되었다. 기존 라벨은 쓰지 않고 새로 라벨링한다. baseline 숫자를 내는 실행(Task 6 이후의 실행 단계)은 새 라벨이 나온 뒤로 미루고, 코드와 형식 준비는 합성 데이터로 테스트하며 진행한다. 새 라벨 전까지 Task 6b의 "실제 실행"(Step 5)과 Task 8의 모델 비교 실행은 하지 않는다.

---

### Task 1: 저장소 뼈대와 문서 기본

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `.github/pull_request_template.md`
- Create: `src/wl_snatch/__init__.py`, `tests/test_scaffold.py`
- Create: `docs/vision.md`, `docs/roadmap.md`, `docs/decisions/0001-environment.md`, `docs/literature/README.md`, `data/README.md`
- Create (빈 디렉터리 유지): `scripts/.gitkeep`, `tutorials/.gitkeep`

**Interfaces:**
- Produces: 설치된 패키지 `wl_snatch`. 이후 모든 Task가 `from wl_snatch... import ...`로 가져온다.

- [ ] **Step 1: 버전 확인.** 공식 문서로 Python, Ultralytics, PyTorch의 현재 안정 버전과 서로의 호환을 확인한다. 결과를 `docs/decisions/0001-environment.md`에 날짜와 출처(URL)와 함께 적는다.
- [ ] **Step 2: 실패하는 테스트 작성** `tests/test_scaffold.py`:
  - `test_package_imports`: `import wl_snatch` 성공.
  - `test_data_dir_is_git_ignored`: `git check-ignore data/raw/anything.mp4`가 0을 반환.
  - `test_weights_are_git_ignored`: `git check-ignore model.pt`가 0을 반환.
- [ ] **Step 3: 실패 확인.** Run: `uv run pytest tests/test_scaffold.py -v` → FAIL (패키지 없음).
- [ ] **Step 4: 뼈대 작성.** `pyproject.toml`은 setuptools src 레이아웃, 패키지명 `wl_snatch`, spec 9절의 라이브러리와 dev 그룹(pytest, jupyterlab). `.gitignore`는 `data/*`(단 `data/README.md` 제외), `*.pt`, `artifacts/`, `.env`, `.venv/`, 노트북 체크포인트를 제외한다. `docs/vision.md`에는 spec 1·2·4절을 옮겨 쓰고 **표현 규칙**(근거 전에 판정 금지, 비유 금지, 측정값과 `N/A`+사유 중심)을 별도 절로 둔다. `docs/roadmap.md`는 G2~G7의 순서와 의존성만 적고 CI를 "첫 PR 이후"로 적는다. `docs/literature/README.md`에는 이 대화에서 찾은 문헌과 URL을 옮기고, 각 항목에 "초록 기준, 원문 미확인" 표기를 붙인다. PR 템플릿에는 표현 규칙 체크박스, "노트북에 로직 없음" 체크박스, 리뷰어가 확인할 단계 간 약속 항목을 넣는다.
- [ ] **Step 5: 통과 확인.** Run: `uv run pytest tests/test_scaffold.py -v` → 3 passed.
- [ ] **Step 6: 커밋.** `main`의 부트스트랩 커밋(`d71a0fe`)은 사용자의 허락으로 이미 올라갔다. 이후 모든 작업은 브랜치 `feature/scaffold`에서 커밋하고, `main`에는 직접 커밋하거나 push하지 않는다. `.gitignore`에 `.superpowers/`와 `.claude/worktrees/`도 추가한다.

### Task 2: 단계 간 약속 `contracts.py`

**Files:**
- Create: `src/wl_snatch/contracts.py`, `docs/contracts/detection.md`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces:
  - `BoxDetection(frame_index: int, x1: float, y1: float, x2: float, y2: float, confidence: float, label: str)` frozen dataclass, 좌표는 픽셀.
  - `FrameDetections(video_id: str, frame_index: int, boxes: tuple[BoxDetection, ...], na_reason: str | None)` frozen dataclass. `na_reason`이 None이 아니면 이 프레임은 `N/A`이며 `boxes`는 비어 있어야 한다.
  - `write_detections_csv(frames: list[FrameDetections], path: Path) -> None`
  - `read_detections_csv(path: Path) -> list[FrameDetections]`

- [ ] **Step 1: 실패하는 테스트 작성.**
  - `test_na_frame_must_have_no_boxes`: `na_reason`이 있는데 박스가 있으면 `ValueError`.
  - `test_csv_roundtrip_preserves_frames`: 박스 있는 프레임, 박스 없는 프레임(탐지 0개, `na_reason=None`), `N/A` 프레임을 쓰고 읽어 동일한지 확인.
  - `test_zero_detections_differs_from_na`: 읽은 뒤에도 "0개 탐지"와 "N/A"가 구분되는지 확인.
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_contracts.py -v` → FAIL.
- [ ] **Step 3: 구현.** CSV 한 행은 박스 하나이고, 박스가 없는 프레임은 좌표가 빈 한 행으로 쓴다(프레임 존재를 보존하기 위함).
- [ ] **Step 4: 통과 확인.** → 3 passed.
- [ ] **Step 5: `docs/contracts/detection.md` 작성.** 형식, 좌표 규칙, `N/A` 규칙, 다음 단계(바 경로)가 이 형식을 입력으로 받는다는 점을 적는다.
- [ ] **Step 6: 커밋.**

### Task 3: 아카이브 데이터 가져오기

**Files:**
- Create: `src/wl_snatch/data_import.py`, `scripts/import_archive_data.py`
- Test: `tests/test_data_import.py`

**Interfaces:**
- Produces:
  - `ImportReport(files_copied: int, videos_copied: int, rewritten_paths: tuple[str, ...])` frozen dataclass.
  - `import_archive(zip_path: Path, data_dir: Path) -> ImportReport`

**동작:** zip의 `raw video/` 폴더(공백 포함)를 `data/raw/videos/`로 복사한다. 이는 `videos.csv`의 `path` 열 형식(`data/raw/videos/<id>.mp4`)과 맞춘 것이다. `metadata/`, `annotations/`, `splits/`, `plate_dataset/`, `plate_dataset_v2/`는 같은 이름으로 `data/` 아래에 둔다. `dataset.yaml`의 `path:`는 다른 사람의 절대경로이므로 가져올 때 상대경로 `.`로 다시 쓴다. 허용된 최상위 폴더 밖의 항목은 복사하지 않고 `skipped` 대신 보고한다. 이미 있는 파일은 덮어쓰지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성** (임시 폴더에 작은 zip을 만든다).
  - `test_raw_video_folder_is_mapped_to_data_raw_videos`
  - `test_dataset_yaml_path_is_rewritten_to_relative`
  - `test_zip_slip_entry_is_rejected`: `../evil.txt` 항목이 있으면 `ValueError`, 아무것도 복사되지 않음.
  - `test_absolute_path_entry_is_rejected`
  - `test_existing_file_is_not_overwritten`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_data_import.py -v` → FAIL.
- [ ] **Step 3: 구현.** 모든 항목의 해석된 경로가 `data_dir` 밑에 있는지 복사 전에 전부 검사한 뒤 복사한다.
- [ ] **Step 4: 통과 확인.** → 5 passed.
- [ ] **Step 5: 실제 실행.** Run: `uv run python scripts/import_archive_data.py "/Users/fabxoe/Downloads/drive-download-20261104T143056Z-1-001.zip"`. 기대: `videos_copied=54`, `data/metadata/videos.csv` 존재, `git status`에 `data/` 파일이 나타나지 않음.
- [ ] **Step 6: 커밋.**

### Task 4: manifest 검증과 사람 단위 분할

**Files:**
- Create: `src/wl_snatch/manifest.py`, `src/wl_snatch/split.py`, `scripts/make_split.py`
- Test: `tests/test_manifest.py`, `tests/test_split.py`

**Interfaces:**
- Consumes: `data/metadata/videos.csv` (열: `video_id, person_id, group, has_plate, path, read_status, ...`).
- Produces:
  - `load_manifest(path: Path) -> pd.DataFrame` — 필수 열이 없으면 `ValueError`(없는 열 이름을 메시지에 포함).
  - `make_person_split(manifest: pd.DataFrame, seed: int, validation_fraction: float) -> pd.DataFrame` — 열 `video_id, person_id, group, role`, `role`은 `train` 또는 `validation`.
  - `assert_no_person_leak(split: pd.DataFrame) -> None` — 같은 `person_id`가 두 역할에 있으면 `AssertionError`.

**동작:** 분할 단위는 `person_id`. 그룹(A/B/C)별로 사람을 나눠서 각 그룹이 가능하면 양쪽에 들어가게 한다. 한 그룹에 사람이 1명뿐이면 그 사람은 `train`에 넣고 보고한다.

- [ ] **Step 1: 실패하는 테스트 작성.**
  - `test_missing_required_column_names_the_column`
  - `test_split_has_no_person_in_both_roles`
  - `test_split_is_deterministic_for_same_seed`
  - `test_person_with_many_videos_stays_in_one_role`
  - `test_group_with_single_person_goes_to_train_and_is_reported`
  - `test_leak_checker_raises_on_leaky_split`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_manifest.py tests/test_split.py -v` → FAIL.
- [ ] **Step 3: 구현.** 정렬된 `person_id`를 seed로 섞고 그룹별 비율로 나눈다. 결과는 `data/splits/person_split.csv`로 저장한다(아카이브의 `split_role` 열은 참고만 하고 쓰지 않는다).
- [ ] **Step 4: 통과 확인.** → 6 passed.
- [ ] **Step 5: 실제 실행.** Run: `uv run python scripts/make_split.py --seed 0 --validation-fraction 0.3`. 기대: 사람 수와 그룹별 인원이 출력되고 `assert_no_person_leak`가 통과한다.
- [ ] **Step 6: 커밋.**

### Task 5: 아카이브 탐지 라벨 점검

**Files:**
- Create: `src/wl_snatch/label_audit.py`, `scripts/audit_plate_labels.py` (구현 당시에는 `scripts/plate_label_contact_sheet.py`와 `contact_sheet.py`도 만들었으나 이미지를 파일로 만드는 코드라 #11, #12에 따라 삭제했다)
- Test: `tests/test_label_audit.py`

**Interfaces:**
- Produces:
  - `LabelIssue(image: str, kind: str, detail: str)` frozen dataclass. `kind`는 `missing_label`, `empty_label`, `out_of_range`, `bad_class`, `orphan_label`, `duplicate_box` 중 하나.
  - `AuditReport(images: int, labeled_boxes: int, issues: tuple[LabelIssue, ...])` frozen dataclass.
  - `audit_yolo_labels(images_dir: Path, labels_dir: Path, allowed_classes: frozenset[int]) -> AuditReport`

- [ ] **Step 1: 실패하는 테스트 작성** (임시 폴더에 가짜 이미지와 라벨).
  - `test_clean_dataset_has_no_issues`
  - `test_image_without_label_file_is_reported`
  - `test_empty_label_file_is_reported`
  - `test_coordinate_outside_zero_one_is_reported`
  - `test_unknown_class_id_is_reported`
  - `test_label_without_image_is_reported`
  - `test_two_identical_boxes_are_reported_as_duplicate`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_label_audit.py -v` → FAIL.
- [ ] **Step 3: 구현.** 라벨은 YOLO 형식(`class cx cy w h`, 0~1)으로 읽는다. 예외를 던지지 않고 `LabelIssue`로 담는다.
- [ ] **Step 4: 통과 확인.** → 7 passed.
- [ ] **Step 5: 실제 점검.** `plate_dataset`(30장)과 `plate_dataset_v2`(150장)에 실행해 `artifacts/label_audit/`에 보고서를 저장한다. (당시에는 컨택트 시트 이미지로 눈 확인을 했다. 이후 방식은 Task 5b의 뷰어로 대체한다.)
- [ ] **Step 6: 커밋.** 이후 **Checkpoint**로 간다.

### Task 5b: 프레임 읽기와 라벨 확인 뷰어 — 박스 (이슈 #12, #11, #10 반영)

**배경:** 얼굴이 담긴 이미지는 저장소에 올리지 않고 파일로도 만들지 않는다(#11). 라벨은 좌표와 프레임 번호만 git에 있고, 확인은 로컬 영상에서 프레임을 메모리로 읽어 좌표를 그린 **배열**을 돌려받아 노트북이 화면에 표시한다(#12). 이 Task는 박스만 다룬다. 관절 점과 단계 점은 형식이 정해진 뒤(#8, #9) 같은 구조에 추가한다. 아래 값은 잠정이다.

**사전 측정(2026-11-06, 읽기 전용, 영상 54개):**
- OpenCV `CAP_PROP_FRAME_COUNT`와 순차 디코딩 수가 49개 영상에서 다르다(컨테이너 값이 큼). ffprobe `-count_frames`는 OpenCV 디코딩 수와 같다(6개 영상). `videos.csv`의 `frame_count`는 디코딩 수와 같다(샘플 6개).
- 타임스탬프 간격은 52개 영상에서 일정(24, 30, 60fps). `a05_sn_f_01`은 처음 두 프레임 사이만 8.7ms. `b00_sn_f_01`은 5.0ms와 18.3ms가 하나씩. `c01_sn_f_02`, `c04_sn_f_03`은 첫 프레임이 66.7ms.
- `videos.csv`의 `fps`는 평균값으로 추정되며 시간 계산에 쓰지 않는다.
- 동료가 프레임 번호를 디코딩 순서(0부터)로 하자는 제안에 동의했다(#10).

**Files:**
- Create: `src/wl_snatch/frames.py`, `src/wl_snatch/label_viewer.py`
- Test: `tests/test_frames.py`, `tests/test_label_viewer.py`
- 재사용: `label_audit.read_boxes`(YOLO 라벨 읽기). 읽는 부분과 그리는 부분을 분리한다.

**Interfaces:**
- Produces (`frames.py`):
  - `read_frame(video_path: Path, frame_index: int) -> np.ndarray`: 처음부터 순차 디코딩해 `frame_index`번째(0부터)를 반환한다. 탐색(seek)을 쓰지 않는다. 읽을 수 없거나 범위를 벗어나면 영상 이름과 번호가 담긴 `ValueError`. 조용히 다른 프레임을 반환하지 않는다.
  - `count_frames(video_path: Path) -> int`: 순차 디코딩으로 센 프레임 수. 컨테이너 메타데이터 값은 쓰지 않는다.
  - `frame_times_ms(video_path: Path) -> list[float]`: 디코딩한 프레임마다의 시각(ms). 첫 프레임을 0으로 맞춘다(첫 프레임이 0이 아닌 영상이 있다).
- Produces (`label_viewer.py`):
  - `Overlay(boxes: tuple[tuple[float, float, float, float], ...], skipped: tuple[str, ...])` frozen dataclass. 박스는 0~1로 정규화한 `cx, cy, w, h`. `skipped`는 좌표가 `N/A`라 건너뛴 항목의 사유.
  - `flag_box_issues(boxes, min_side: float = 0.02, max_side: float = 0.9) -> list[str]`: 한 변이 `min_side`보다 작거나 `max_side`보다 큰 박스의 설명. 기준값은 인자로 바꾼다.
  - `draw_overlay(frame: np.ndarray, overlay: Overlay) -> np.ndarray`: 입력을 바꾸지 않고 복사본에 박스를 그려 **배열을 돌려준다.**
  - `render_label_frame(video_path: Path, frame_index: int, overlay: Overlay) -> tuple[np.ndarray, list[str]]`: 프레임을 읽어 그린 배열과 안내 문구(이상값 설명, 건너뛴 사유)를 돌려준다. 화면 표시와 파일 저장은 하지 않는다.
- 이 모듈에는 이미지를 파일로 저장하는 함수를 두지 않는다. 화면 표시는 노트북이 한다.

- [ ] **Step 1: 실패하는 테스트 작성** (OpenCV로 임시 폴더에 프레임마다 색이 다른 5프레임 합성 영상을 만든다. 실제 사람이 나온 영상은 쓰지 않는다).
  - `test_read_frame_returns_requested_index`
  - `test_read_frame_out_of_range_raises_value_error_with_name_and_index`
  - `test_read_frame_missing_video_raises_value_error`
  - `test_count_frames_matches_decoded_frames`
  - `test_frame_times_start_at_zero_and_have_one_entry_per_frame`
  - `test_draw_overlay_does_not_modify_input`
  - `test_draw_overlay_changes_pixels_at_box_edge`
  - `test_flag_box_issues_flags_width_0_024_height_0_0047`
  - `test_flag_box_issues_passes_normal_plate_box`
  - `test_flag_box_issues_threshold_is_configurable`
  - `test_render_label_frame_reports_skipped_reasons`
  - `test_no_image_file_is_written` (렌더링 후 작업 폴더와 임시 폴더에 새 파일이 없음)
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_frames.py tests/test_label_viewer.py -v` → FAIL.
- [ ] **Step 3: 구현.** 프레임 읽기는 순차 디코딩.
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 실제 확인.** 로컬 영상 54개에서 `count_frames`가 `videos.csv`의 `frame_count`와 같은지 비교해 표로 남긴다(표에는 영상 id와 숫자만). 다르면 사유를 기록한다.
- [ ] **Step 6: 커밋.**

**상태:** 프레임 번호 기준(#10)이 사실상 합의되어 구현할 수 있다. 박스 형식은 YOLO로 가정하며, 새 라벨 형식이 정해지면 읽는 부분만 맞춘다.

### Task 5c: 라벨링 노트북 — 점 라벨 (이슈 #8, #12)

**상태:** 점 라벨 형식은 도메인 담당이 #12(2026-11-11)에서 확정했다. 점마다의 허용 오차 크기와 기준 pass 선택은 시범 라벨링 뒤에 정하며 이 Task와 무관하다.

**배경:** 도메인 담당이 주피터 노트북에서 직접 점을 찍는다. 노트북은 함수와 클래스를 정의하지 못하고 셀은 15줄 이하이므로(노트북 규칙), 위젯 연결을 포함한 모든 동작은 `src/`의 클래스가 맡고 노트북은 객체를 만들어 호출만 한다. 화면에 보이는 이미지는 메모리에서만 인코딩하고 파일로 저장하지 않는다(#11).

**확정된 규칙 (#12, 2026-11-11):**
- 형식: `video_id, event, frame, pass`. event는 `ready, transition, drive, catch, finish`. pass는 1 또는 2.
- 덮어쓰기: `(video_id, event, pass)`가 모두 같을 때만 새 값이 옛 값을 대체한다.
- pass 1과 pass 2는 별개 기록이다. 같은 프레임을 찍어도 둘 다 보존한다.
- pass 2가 진행 중인 영상은 pass 1을 수정할 수 없다. (구현 규칙, 컨트롤러 판단: 같은 `video_id`의 pass 2 기록이 하나라도 있으면 그 영상의 pass 1 기록은 추가와 수정이 모두 `ValueError`.)
- half 영상(`completion == "h"`)은 ready, transition, drive만 필수. catch와 finish는 "ignore"이고 누락과 구별한다. (구현 규칙, 컨트롤러 판단: ignore는 데이터에 행으로 두지 않고 `completion` 값에서 도출한다. half 영상에 catch, finish를 기록하려 하면 `ValueError`. 누락은 필수 event 중 기록이 없는 것이다.)

**Files:**
- Create: `src/wl_snatch/labeling.py`, `tutorials/label_points.ipynb`, `.pre-commit-config.yaml`
- Create: `tests/test_labeling.py`, `tests/test_notebook_hygiene.py` (Task 9에서 이쪽으로 옮김)
- Modify: `pyproject.toml`과 `uv.lock`(`ipywidgets` 의존성 추가, dev 그룹에 `nbstripout`), `data/README.md`(라벨 파일 위치와 규칙 한 단락)

**Interfaces:**
- Consumes: `frames.read_frame`, `frames.count_frames`.
- Produces (`labeling.py`):
  - `EVENTS = ("ready", "transition", "drive", "catch", "finish")`
  - `required_events(completion: str) -> tuple[str, ...]`: `"f"`는 5개, `"h"`는 앞의 3개. 다른 값은 `ValueError`.
  - `ignored_events(completion: str) -> tuple[str, ...]`: `"f"`는 빈 튜플, `"h"`는 `("catch", "finish")`.
  - `PointLabel(video_id: str, event: str, frame: int, pass_no: int)` frozen dataclass.
  - `PointLabelStore(path: Path)`:
    - `add(video_id, event, frame, pass_no, *, completion, frame_count) -> None`: event가 `EVENTS`에 없으면, pass_no가 1이나 2가 아니면, `frame`이 `[0, frame_count)` 밖이면, half 영상의 ignore event이면, pass 2 기록이 있는 영상의 pass 1이면 `ValueError`(메시지에 video_id와 값). 같은 `(video_id, event, pass_no)`면 대체. 파일은 임시 파일에 쓴 뒤 교체한다(쓰다 멈춰도 기존 파일이 남는다). 행은 `(video_id, pass, EVENTS 순서)`로 정렬.
    - `read(video_id: str | None = None, pass_no: int | None = None) -> list[PointLabel]`: 파일이 없으면 빈 목록. 형식이 맞지 않는 행이 있으면 줄 번호가 담긴 `ValueError`.
    - `missing_events(video_id: str, completion: str, pass_no: int) -> tuple[str, ...]`: 필수 event 중 그 pass에 기록이 없는 것. ignore event는 포함하지 않는다.
  - `shuffled_order(video_ids: Sequence[str], seed: int) -> list[str]`: 같은 seed면 같은 순서이고 입력의 순열.
  - `LabelingSession(video_path: Path, video_id: str, store: PointLabelStore, pass_no: int, completion: str)`:
    - `frame_count: int`(생성 시 `count_frames`), `frame: int`(현재, 0에서 시작)
    - `goto(frame: int) -> None`: 범위 밖이면 `ValueError`. `step(delta: int) -> None`: 첫·마지막 프레임에서 멈춘다(오류 없음).
    - `record(event: str) -> None`: 현재 프레임을 `store.add`로 기록.
    - `visible_labels() -> list[PointLabel]`: 이 세션의 pass 기록만. pass 2 세션에는 pass 1 기록이 보이지 않는다.
    - `view() -> tuple[np.ndarray, list[str]]`: 현재 프레임 배열(그린 글자 없이 원본)과 안내 문구(`frame i / n`, 이 pass에서 기록한 event와 프레임, 누락된 필수 event, ignore event).
    - `show() -> None`: `ipywidgets`로 슬라이더, 앞뒤 버튼, event 선택, 기록 버튼을 만들어 표시한다. 이미지는 `cv2.imencode`로 메모리에서만 인코딩해 `ipywidgets.Image`에 넣고 파일로 저장하지 않는다. 위젯 연결(콜백)은 모두 이 메서드 안에서만 만든다.

- [ ] **Step 1: 실패하는 테스트 작성** (합성 영상과 합성 숫자만 사용. 위젯 UI가 아니라 메서드를 검증한다. 실제 사람이 나온 영상은 쓰지 않는다).
  - `test_recorded_point_is_saved_and_read_back_equal`
  - `test_recording_same_video_event_pass_overwrites`
  - `test_pass_one_and_pass_two_are_kept_separately_even_for_same_frame`
  - `test_pass_one_cannot_be_added_or_changed_after_pass_two_exists_for_that_video`
  - `test_pass_one_of_another_video_is_unaffected_by_pass_two`
  - `test_unknown_event_raises_value_error`
  - `test_pass_other_than_one_or_two_raises_value_error`
  - `test_frame_outside_video_raises_value_error`
  - `test_half_video_rejects_catch_and_finish`
  - `test_required_events_full_and_half` / `test_ignored_events_full_and_half` / `test_unknown_completion_raises_value_error`
  - `test_missing_events_lists_only_required_events_without_a_record`
  - `test_malformed_row_raises_value_error_with_line_number`
  - `test_failed_add_leaves_existing_file_unchanged`
  - `test_shuffled_order_is_deterministic_for_seed_and_a_permutation`
  - `test_step_clamps_at_first_and_last_frame` / `test_goto_out_of_range_raises_value_error`
  - `test_pass_two_session_hides_pass_one_labels`
  - `test_view_text_reports_missing_and_ignored_events`
  - `test_session_and_store_write_no_image_file` (CSV 외에 새 파일이 없음)
  - `tests/test_notebook_hygiene.py`의 5개(Task 9의 목록): `test_no_function_or_class_definitions_in_notebooks`, `test_no_test_or_assert_code_in_notebooks`, `test_notebook_code_cells_are_short`, `test_notebooks_have_no_stored_outputs`, `test_test_suite_never_executes_notebooks`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_labeling.py tests/test_notebook_hygiene.py -v` → FAIL. 위생 테스트는 노트북 추가 전에 일부러 `def`가 든 임시 노트북으로 실패를 한 번 확인하고 지운다.
- [ ] **Step 3: 구현.** `ipywidgets`는 `show()` 안에서 가져온다(위젯 없이도 나머지가 동작하도록).
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 노트북 작성.** `tutorials/label_points.ipynb`는 마크다운 안내 셀과, 영상 id·경로·pass 번호·completion을 넣고 `LabelingSession(...).show()`를 호출하는 셀만 갖는다. 저장된 출력이 없다. `.pre-commit-config.yaml`에 `nbstripout` 훅을 둔다. 노트북은 실행하지 않는다(테스트가 실행하지 않는다).
- [ ] **Step 6: `data/README.md`에 한 단락 추가.** 라벨 파일 위치(`data/annotations/point_labels.csv`), 열, 덮어쓰기와 pass 규칙, half 영상 규칙.
- [ ] **Step 7: 커밋.**

**이 Task에서 제외:** 관절 점 라벨(#9), 두 번 찍은 결과의 비교 그림과 편차 계산(시범 라벨링 뒤 별도 Task), 노트북 자동 실행.

### Task 5d: 영상 시간 간격 표 (이슈 #10)

**배경:** 54개 영상의 해상도, 프레임 수, 프레임 간격(최소·중앙값·최대), 첫 프레임 시각, 시간 이상 표시(`timestamp_note`), 이상 간격의 프레임 번호를 표 하나로 남긴다. 동료(도메인 담당)가 이상 간격의 위치를 보고 영상 포함 여부를 판단한다. 표에는 영상 id와 숫자만 들어가므로 개인정보가 없고 `data/metadata/`에서 git이 추적한다. 아래 기준값은 잠정이며 동료가 확인한다.

**Files:**
- Modify: `src/wl_snatch/frames.py` (`raw_frame_times_ms`, `frame_size` 추가, `frame_times_ms`가 `raw_frame_times_ms`를 쓰도록 정리)
- Create: `src/wl_snatch/video_timing.py`, `scripts/video_timing_table.py`
- Test: `tests/test_video_timing.py`, `tests/test_frames.py`에 추가
- 산출물: `data/metadata/video_timing.csv`

**Interfaces:**
- Consumes: `frames.py`의 순차 디코딩 규칙. `data/metadata/videos.csv`의 `video_id`, `path` 열(`path`는 저장소 루트 기준).
- Produces (`frames.py`):
  - `raw_frame_times_ms(video_path: Path) -> list[float]`: 디코더가 보고한 시각(ms)을 이동 없이 돌려준다. 모든 시각이 같으면(프레임 2개 이상) `frame_times_ms`와 같은 `ValueError`.
  - `frame_size(video_path: Path) -> tuple[int, int]`: 첫 디코딩 프레임의 `(width, height)`. 프레임이 없으면 영상 이름이 담긴 `ValueError`.
- Produces (`video_timing.py`):
  - `TimingRow` frozen dataclass: `video_id: str`, `width: int | None`, `height: int | None`, `frames: int | None`, `fps_from_median_gap: float | None`, `gap_min_ms: float | None`, `gap_median_ms: float | None`, `gap_max_ms: float | None`, `first_frame_ms: float | None`, `timestamp_note: str`, `irregular_gap_frames: tuple[int, ...]`.
  - `analyze_timing(video_id: str, width: int, height: int, raw_times_ms: Sequence[float], offset_tolerance_ms: float = 0.5, gap_tolerance_ms: float = 2.5, gap_tolerance_fraction: float = 0.10) -> TimingRow`
    - 간격 `gap[g] = t[g+1] - t[g]`. 중앙값에서 `max(gap_tolerance_ms, gap_tolerance_fraction * median)`를 넘게 벗어난 간격은 이상이며, **프레임 번호 `g`(프레임 g와 g+1 사이)를 `irregular_gap_frames`에 기록한다.**
    - `first_frame_ms = raw_times_ms[0]`이고 `offset_tolerance_ms`보다 크면 `first_frame_offset`.
    - `timestamp_note`: 해당하는 값을 `;`로 이어 쓴다(`first_frame_offset`, `irregular_gap` 순). 없으면 빈 문자열.
    - 프레임이 2개 미만이면 간격 관련 값은 `None`, `timestamp_note`는 `N/A: fewer than 2 frames`.
    - `fps_from_median_gap = 1000 / gap_median_ms`.
  - `unreadable_row(video_id: str, reason: str) -> TimingRow`: 숫자 필드는 `None`, `timestamp_note`는 `N/A: <reason>`.
  - `write_timing_csv(rows: Sequence[TimingRow], path: Path) -> None`: 열 순서 `video_id,width,height,frames,fps_from_median_gap,gap_min_ms,gap_median_ms,gap_max_ms,first_frame_ms,timestamp_note,irregular_gap_frames`. `None`은 빈 칸, `irregular_gap_frames`는 `;`로 이은 번호. 숫자는 소수 둘째 자리로 쓴다. 부모 폴더는 만든다.
- 이 모듈과 스크립트는 이미지를 만들거나 저장하지 않는다. 출력은 CSV와 텍스트 요약뿐이다.

- [ ] **Step 1: 실패하는 테스트 작성** (시각은 합성 숫자 목록으로 만든다. 실제 영상은 쓰지 않는다. `frames.py` 추가분만 `make_video` 합성 영상을 쓴다).
  - `test_constant_30fps_has_no_note_and_matches_statistics`
  - `test_first_frame_offset_is_noted`
  - `test_small_first_frame_time_within_tolerance_is_not_noted`
  - `test_irregular_gap_is_noted_with_frame_number` (간격 하나가 8.7ms인 24fps 목록: 해당 `g`가 기록됨)
  - `test_timebase_rounding_at_60fps_is_not_flagged` (16.7, 16.7, 18.3 간격은 이상이 아님)
  - `test_large_deviation_at_60fps_is_flagged` (5.0ms 간격은 이상)
  - `test_notes_join_both_with_semicolon_in_fixed_order`
  - `test_fewer_than_two_frames_gives_na_note`
  - `test_unreadable_row_has_na_note_and_empty_numbers`
  - `test_write_timing_csv_header_and_values` (헤더 순서, `None`은 빈 칸, 번호는 `;`, 소수 둘째 자리)
  - `test_frame_size_returns_width_height_of_decoded_frame` / `test_frame_size_empty_video_raises_value_error` (frames 테스트)
  - `test_raw_frame_times_not_shifted_and_frame_times_are_shifted_copy` (frames 테스트)
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_video_timing.py tests/test_frames.py -v` → FAIL.
- [ ] **Step 3: 구현.** `scripts/video_timing_table.py --data-dir data --out data/metadata/video_timing.csv`는 `videos.csv`의 영상마다 행을 만든다. 읽을 수 없는 영상은 `unreadable_row`로 남기고 중단하지 않는다. 끝에 영상 수, `timestamp_note`별 개수, 읽지 못한 영상 수를 텍스트로 출력한다.
- [ ] **Step 4: 통과 확인.** `uv run pytest -q` 전체 통과, `uv run ruff check`.
- [ ] **Step 5: 실제 실행.** 스크립트를 로컬 54개 영상에 실행해 `data/metadata/video_timing.csv`를 만든다. 표에서 다음을 보고서에 적는다: `timestamp_note`별 영상 수, 이상 간격이 있는 영상과 번호, 앞서 `ffprobe`로 확인한 영상(`a05_sn_f_01`, `b00_sn_f_01`, `c01_sn_f_02`, `c04_sn_f_03`)과 이 표의 결과가 일치하는지. 일치하지 않으면 사유를 적는다.
- [ ] **Step 6: 커밋.** 표 CSV 포함. 이미지 파일은 만들지 않는다.

**기준값(잠정):** 간격 이상 허용 = `max(2.5ms, 중앙값의 10%)`, 첫 프레임 시각 허용 = 0.5ms. 24fps와 30fps의 정상 흔들림은 1.7ms 이내이고 60fps 영상의 타임베이스 반올림 단위가 약 1.67ms라는 사전 측정에 맞춘 값이다. 동료가 확인한다.

### Task 5e: 원판 박스 라벨링 노트북 (이슈 #2)

**상태:** 규칙은 도메인 담당이 #2(2026-11-12~13)에서 확정했다. 의존성 `ipyevents` 추가에도 동의했다. 시범 검수 15장(`a01_sn_f_01`, `b01_sn_f_01`, `c02_sn_f_01`의 재판독 반영 후 pass 1 프레임 5장씩)은 도메인 담당이 이 도구로 한다.

**배경:** 도메인 담당이 로컬 영상에서 지정한 프레임을 열고, 카메라에 가까운 원판 하나의 보이는 외곽에 박스를 그린다. 노트북은 호출만 하므로(노트북 규칙) 모든 동작은 `src/`의 클래스가 맡는다. 화면의 이미지는 메모리에서만 인코딩하고 파일로 저장하지 않는다(#11).

**확정된 규칙 (#2):**
- 박스 대상은 카메라에 가까운 쪽 원판 하나이고, 보이는 외곽만 감싼다. 일부 가려지거나 화면 밖으로 잘렸으면 보이는 부분만 감싸고 화면 경계를 넘지 않게 그린다. 가려진 부분을 추정해 넓히지 않는다.
- 저장: 라벨 하나당 파일 하나 `data/annotations/plate_boxes/<video_id>_f<프레임>.txt`. 한 줄 `class cx cy w h`(YOLO, 0~1 정규화). 클래스는 하나이므로 `class`는 0.
- 빈 `.txt`는 "확인했으나 가까운 쪽 원판 없음"이고, 파일이 없으면 "아직 라벨링 안 함"이다. 사유 열은 두지 않는다(#2에서 사유 방식은 합의하지 않았고 빈 파일 방식에만 동의했다).
- 추출 범위 예외는 `data/annotations/plate_frame_ranges.csv`(열 `video_id,start_frame,end_frame,reason`)에 예외가 있는 영상만 적는다. 범위 밖 프레임에는 박스도 빈 라벨도 만들지 않는다. 이 Task에서 `b02_sn_h_02,0,53,plate_leaves_frame` 한 줄로 파일을 만든다.
- 프레임 번호는 디코딩 순서 0부터.
- 입력: 마우스 드래그가 기본이고, 드래그가 작동하지 않을 때를 위해 슬라이더로 좌상단과 우하단을 조정할 수 있어야 한다.

**제 판단으로 정한 것 (컨트롤러 판단, 도메인 담당이 바꿀 수 있다):**
- 같은 프레임에 다시 저장하면 새 값이 옛 값을 대체한다(박스를 빈 파일로, 빈 파일을 박스로 바꾸는 경우 포함).
- 박스는 `label_viewer.NormBox`(`cx, cy, w, h`)를 그대로 쓴다. 숫자는 소수 여섯째 자리까지 쓴다.
- 박스가 화면 경계로 잘린 뒤 너비나 높이가 0 이하이면 저장하지 않고 `ValueError`.

**Files:**
- Create: `src/wl_snatch/plate_boxes.py`, `tutorials/label_plates.ipynb`, `data/annotations/plate_frame_ranges.csv`
- Create: `tests/test_plate_boxes.py`
- Modify: `src/wl_snatch/label_viewer.py`(`draw_overlay`에 색 인자 추가), `tests/test_label_viewer.py`, `tests/test_scaffold.py`(`plate_boxes/*.txt`가 추적되는지), `pyproject.toml`과 `uv.lock`(`ipyevents` 추가), `data/README.md`(라벨 위치와 규칙 한 단락)

**Interfaces:**
- Consumes: `frames.read_frame`, `frames.count_frames`, `frames.frame_size`(원본 픽셀 표시용), `label_viewer.NormBox`, `label_viewer.draw_overlay`, `label_viewer.flag_box_issues`, `labeling.guidance_html`.
- Produces (`label_viewer.py`): `draw_overlay(frame, overlay, color=BOX_COLOR_BGR)`. 기본값일 때 동작은 그대로다.
- Produces (`plate_boxes.py`):
  - `corners_to_norm_box(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> NormBox`: 모서리 순서는 상관없다. 좌표를 `[0, width] x [0, height]`로 자른 뒤 정규화한다. 자른 뒤 너비나 높이가 0 이하이면 `ValueError`. 화면 좌표(`width`, `height`는 화면에 보이는 크기)와 원본 픽셀 좌표(원본 크기)에 같은 함수를 쓰며, 비례하는 값이면 같은 결과가 나온다.
  - `format_yolo_line(box: NormBox) -> str`: `"0 cx cy w h"`, 소수 여섯째 자리, 줄바꿈 없음.
  - `parse_yolo_line(text: str, source: str) -> NormBox`: 필드가 5개가 아니거나, class가 0이 아니거나, 값이 `[0, 1]` 밖이거나, w나 h가 0 이하이면 `source`가 담긴 `ValueError`.
  - `FrameRange` frozen dataclass: `start_frame: int`, `end_frame: int`, `reason: str`. `read_frame_ranges(path: Path) -> dict[str, FrameRange]`: 파일이 없으면 빈 dict. 열 이름이 다르거나, 정수가 아니거나, `start_frame > end_frame`이거나, `video_id`가 비었거나 중복이면 줄 번호가 담긴 `ValueError`.
  - `PlateBoxStore(boxes_dir: Path, ranges_path: Path | None = None)`:
    - `path_for(video_id: str, frame: int) -> Path`: `<boxes_dir>/<video_id>_f<frame>.txt`.
    - `save_box(video_id, frame, box: NormBox, *, frame_count: int) -> None` / `save_empty(video_id, frame, *, frame_count: int) -> None`: `frame`이 `[0, frame_count)` 밖이거나, 그 영상에 범위 예외가 있고 `[start_frame, end_frame]` 밖이거나, `video_id`가 비었거나, `frame`이 int가 아니면(`bool` 포함) `ValueError`(메시지에 video_id와 값). 파일은 임시 파일에 쓴 뒤 교체한다. 실패하면 기존 파일이 그대로다. `boxes_dir`는 없으면 만든다.
    - `status(video_id, frame) -> str`: `"unlabeled"`(파일 없음), `"empty"`(0바이트 또는 공백만), `"box"`.
    - `read(video_id, frame) -> NormBox | None`: 박스가 있으면 박스, 빈 파일이나 파일 없음이면 `None`. 줄이 둘 이상이거나 형식이 맞지 않으면 파일 이름이 담긴 `ValueError`.
  - `BoxLabelingSession(video_path: Path, video_id: str, store: PlateBoxStore, *, display_width: int = DISPLAY_WIDTH)` (`DISPLAY_WIDTH = 360`, 양의 int가 아니면 ValueError):
    - `frame_count: int`, `frame: int`(0에서 시작), `goto(frame)`(범위 밖이면 `ValueError`), `step(delta)`(양 끝에서 멈춤).
    - `draft: NormBox | None`, `set_draft_from_corners(x1, y1, x2, y2, width, height) -> None`(`corners_to_norm_box`로 계산해 `draft`에 둔다), `clear_draft() -> None`. 프레임을 옮기면 `draft`는 지워진다.
    - `save_box() -> None`: `draft`를 저장. `draft`가 없으면 `ValueError`. `save_empty() -> None`.
    - `view() -> tuple[np.ndarray, list[str]]`: 현재 프레임에 저장된 박스(녹색)와 `draft`(노란색)를 그린 배열, 안내 문구(`frame i / n`, 이 프레임의 `status`, 범위 예외가 있으면 허용 범위와 사유, 박스 크기 경고는 `flag_box_issues`).
    - `pointer_text: str`: 마우스 위치 안내 한 줄. 마우스가 이미지 위에 없으면 빈 문자열.
    - `handle_pointer_event(event: Mapping[str, Any]) -> None`: `ipyevents`가 넘기는 dict를 처리한다. 필요한 키는 `type`, `relativeX`, `relativeY`, `boundingRectWidth`, `boundingRectHeight`. `boundingRectWidth`나 `boundingRectHeight`가 0 이하이면 무시한다.
      - `mousemove`: `pointer_text`를 갱신한다(`x`, `y`는 0~1 비율, 원본 픽셀, 소수 셋째 자리와 정수). 드래그 상태는 바꾸지 않는다.
      - `mousedown`: 시작점을 기억한다.
      - `mouseup`: 시작점이 있으면 `set_draft_from_corners(시작, 끝, boundingRectWidth, boundingRectHeight)`를 호출하고 시작점을 지운다. 시작점이 없으면 무시한다.
      - 그 밖의 `type`은 무시한다.
    - `show() -> None`: `ipywidgets`로 `Image`(화면 폭은 `display_width`, 기본 360px, 높이는 `round(display_width * 높이 / 폭)`), 프레임 슬라이더, 앞뒤 버튼, 좌상단·우하단 슬라이더 4개(정규화 0~1), `save box`와 `save empty` 버튼, `pointer_text`를 보여 줄 `HTML` 한 줄을 만든다. `ipyevents.Event`는 아래 구성으로 `Image`에 붙이고 `on_dom_event`에 `handle_pointer_event`를 연결한다: `watched_events=["mousedown", "mouseup", "mousemove"]`, `prevent_default_action=True`, `throttle_or_debounce="throttle"`, `wait=50`. 슬라이더와 드래그는 같은 `draft`를 갱신한다. `ipywidgets`와 `ipyevents`는 `show()` 안에서 가져온다. 이미지는 `cv2.imencode`로 메모리에서만 인코딩한다.
      - 이 구성은 합성 이미지 시험 노트북으로 실제 Chrome(헤드리스, 화면 배율 1배와 2배)에서 검증했다: 호버, 모서리 클릭, 기준 사각형 드래그, 스크롤한 상태의 드래그가 3개 해상도(720×1280, 1080×1920, 404×720)에서 모두 맞았고 드래그 중 이미지 끌기와 텍스트 선택은 발생하지 않았다. 다른 브라우저는 시험하지 않았고 도메인 담당이 필요 없다고 했다(2026-11-17).

- [ ] **Step 1: 실패하는 테스트 작성** (합성 영상과 합성 숫자만. 위젯과 브라우저가 아니라 메서드를 검증한다. 실제 사람이 나온 영상은 쓰지 않는다).
  - `test_corners_to_norm_box_matches_hand_computed_values` / `test_corner_order_does_not_matter`
  - `test_corners_to_norm_box_same_for_proportional_display_and_original_size` (360x640 화면과 1080x1920 원본)
  - `test_corners_are_clipped_to_frame_and_box_never_leaves_zero_to_one`
  - `test_box_with_no_area_after_clipping_raises_value_error` (완전히 화면 밖, 너비 0)
  - `test_format_yolo_line_has_class_zero_and_six_decimals` / `test_parse_yolo_line_round_trips_and_rejects_bad_lines`
  - `test_saved_box_is_read_back_equal_within_six_decimals`
  - `test_status_unlabeled_empty_and_box`
  - `test_empty_file_and_missing_file_are_different_states`
  - `test_saving_again_replaces_box_with_empty_and_empty_with_box`
  - `test_frame_outside_video_raises_value_error`, `test_bool_or_float_frame_raises_value_error`, `test_blank_video_id_raises_value_error`
  - `test_failed_save_leaves_existing_file_unchanged`
  - `test_frame_outside_range_exception_is_rejected_for_box_and_empty` / `test_frame_at_range_edges_is_accepted` (`end_frame` 포함)
  - `test_video_without_range_exception_accepts_any_frame_in_video`
  - `test_read_frame_ranges_valid_and_missing_file`, `test_read_frame_ranges_rejects_bad_rows_with_line_number` (비정수, start>end, 중복, 빈 id)
  - `test_session_step_clamps_and_goto_out_of_range_raises`
  - `test_session_move_clears_draft`
  - `test_session_save_without_draft_raises`
  - `test_pointer_drag_events_set_draft_equal_to_corner_computation` (mousedown, mousemove, mouseup 순서의 모의 이벤트. 화면 360×640과 원본 비례 크기에서 같은 `draft`)
  - `test_pointer_mouseup_without_mousedown_is_ignored`
  - `test_pointer_mousemove_updates_text_but_not_draft`
  - `test_pointer_event_with_zero_bounding_rect_is_ignored`
  - `test_pointer_unknown_event_type_is_ignored`
  - `test_pointer_text_shows_fractions_and_original_pixels` (원본 크기 기준 픽셀)
  - `test_session_view_draws_saved_box_green_and_draft_yellow` (픽셀 값으로 확인)
  - `test_session_view_text_reports_status_and_range_exception`
  - `test_session_and_store_write_no_image_file` (`.txt` 외 새 파일 없음)
  - `test_draw_overlay_default_color_unchanged` (`test_label_viewer.py`), `test_plate_box_txt_is_tracked_by_gitignore_rules` (`test_scaffold.py`)
  - 노트북 위생 테스트가 새 노트북에도 통과하는지(기존 5개 테스트가 `tutorials/`의 모든 노트북을 본다. 보지 않으면 확장).
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_plate_boxes.py -v` → FAIL.
- [ ] **Step 3: 구현.** `ipyevents`는 `uv add ipyevents`로 추가한다(`pyproject.toml`과 `uv.lock`이 바뀐다). 의존성 추가 전에 설치된 `ipywidgets` 8.1.9와 충돌하지 않는지 `uv lock`이 통과하는지로 확인한다.
- [ ] **Step 4: 통과 확인.** `uv run pytest -q` 전체 통과, `uv run ruff check`.
- [ ] **Step 5: 노트북 작성.** `tutorials/label_plates.ipynb`는 마크다운 안내 셀과, 영상 id·경로를 넣고 `BoxLabelingSession(...).show()`를 호출하는 셀만 갖는다. 저장된 출력이 없다. 실행하지 않는다.
- [ ] **Step 6: `data/annotations/plate_frame_ranges.csv` 생성**(`b02_sn_h_02,0,53,plate_leaves_frame` 한 줄과 헤더). `data/README.md`에 한 단락 추가(저장 위치, 한 줄 형식, 빈 파일과 파일 없음, 범위 예외).
- [ ] **Step 7: 커밋.**

**검증되지 않는 부분 (보고서에 명시):** 실제 JupyterLab에서 마우스 이벤트가 서버까지 전달되는지, 화면 좌표가 원본 좌표와 맞는지는 이 환경에서 시험할 수 없다. 서버 쪽 변환(`corners_to_norm_box`)과 저장은 합성 숫자로 검증하고, 브라우저 동작은 도메인 담당이 `a01_sn_f_01`의 3번 프레임에 박스를 그려 저장한 뒤 다시 열어 원판 외곽과 맞는지로 확인한다(#2, 2026-11-13).

**이 Task에서 제외:** 박스로부터 학습 데이터셋을 만드는 일(Task 6b가 이 형식을 읽도록 바꾼다), 음성 샘플(`b05_sn_f_01`, `b06_sn_f_01`)의 프레임 선정 규칙, 첫 라벨링의 인원·영상 수 결정, 노트북 자동 실행.

### Task 6: 탐지 평가기

**Files:**
- Create: `src/wl_snatch/eval_detection.py`
- Test: `tests/test_eval_detection.py`

**Interfaces:**
- Consumes: `FrameDetections`, `BoxDetection` (Task 2).
- Produces:
  - `GroundTruthFrame(video_id: str, frame_index: int, boxes: tuple[tuple[float, float, float, float], ...])` frozen dataclass. 정의는 `contracts.py`에 있고(A단계 최종 리뷰에서 평가기에서 옮김) `eval_detection`에서도 가져올 수 있다. `boxes`의 각 항목은 `(x1, y1, x2, y2)` 픽셀 좌표이고, `boxes`가 비어 있으면 원판이 없는 프레임이다(예측은 모두 오탐).
  - `DetectionMetrics(precision: float | None, recall: float | None, false_positives_per_frame: float | None, frames_evaluated: int, frames_na: int, na_reasons: dict[str, int], true_positives: int = 0, false_positives: int = 0, false_negatives: int = 0)`. 세 개수는 표본 크기(N)를 보고하기 위한 것이다.
  - `evaluate_detections(predictions: list[FrameDetections], truths: list[GroundTruthFrame], iou_threshold: float) -> DetectionMetrics`

**동작:** 오탐은 정답과 IoU가 임계값 미만이거나 이미 매칭된 정답에 또 매칭된 예측이다. 정답 박스가 없는 프레임(원판 없음)의 예측은 모두 오탐이다. `N/A` 프레임은 평균에서 빼고 사유별로 센다. 분모가 0이면 해당 지표는 `None`이다.

**구현에서 정한 규칙 (컨트롤러 판단):** `truths`에 없는 프레임의 예측은 무시한다(평가는 라벨된 프레임에서만 한다). 정답 프레임에 예측 항목이 없으면 `N/A`(사유 `prediction_missing`)이다. `(video_id, frame_index)`가 `predictions`나 `truths`에서 중복되면 `ValueError`, `iou_threshold`는 `0 < x <= 1`(불리언 제외). 같은 신뢰도의 예측은 입력 순서로 매칭한다. 예측이 `N/A`인 프레임의 정답 박스는 미검출로 세지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성.**
  - `test_perfect_prediction_gives_precision_and_recall_one`
  - `test_box_below_iou_threshold_counts_as_false_positive_and_miss`
  - `test_duplicate_prediction_on_same_truth_counts_one_false_positive`
  - `test_box_on_frame_without_plate_is_false_positive`
  - `test_na_frames_are_excluded_and_counted_by_reason`
  - `test_no_truth_boxes_anywhere_gives_recall_none`
  - `test_no_predictions_gives_precision_none`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_eval_detection.py -v` → FAIL.
- [ ] **Step 3: 구현.** 확신도가 높은 예측부터 탐욕적으로 매칭한다.
- [ ] **Step 4: 통과 확인.** → 계획한 7개와 위 규칙을 확인하는 테스트가 모두 통과한다(예: 신뢰도 내림차순 매칭을 고정하는 `test_matching_order_follows_descending_confidence`).
- [ ] **Step 5: 커밋.**

### Task 6a: 정답 만들기 — 영상 + 원판 박스 라벨 + 분할 (A단계)

**배경:** 평가기(Task 6)와 데이터셋 내보내기(Task 6b)가 같은 정답을 쓰도록, 라벨 파일과 사람 단위 분할에서 `GroundTruthFrame` 목록을 만드는 한 곳을 둔다. 입력은 `data/annotations/plate_boxes/<video_id>_f<프레임>.txt`(정규화 `0 cx cy w h` 또는 빈 파일), `data/metadata/videos.csv`(영상 크기와 프레임 수), `data/splits/person_split.csv`(`role`)이다. 이미지는 읽지도 만들지도 않는다. 이 Task는 이슈 #2의 이미지 정책 결정과 무관하다.

**Files:**
- Create: `src/wl_snatch/ground_truth.py`
- Test: `tests/test_ground_truth.py`

**Interfaces:**
- Consumes: `eval_detection.GroundTruthFrame`(Task 6), `plate_boxes.PlateBoxStore`(`status`, `read`), `manifest.load_manifest`가 돌려주는 표(`video_id`, `width`, `height`, `frame_count`)와 `person_split.csv`의 열(`video_id`, `role`).
- Produces (`ground_truth.py`):
  - `ROLES = ("train", "validation")`
  - `parse_label_name(name: str) -> tuple[str, int]`: `a01_sn_f_01_f3.txt`처럼 `<video_id>_f<숫자>.txt`를 `("a01_sn_f_01", 3)`으로. 형식이 다르면 파일 이름이 담긴 `ValueError`. 영상 id에 `_f`가 들어 있어도 마지막 `_f<숫자>`만 프레임으로 읽는다(정규식 전체 일치).
  - `GroundTruthReport` frozen dataclass: `frames: int`, `with_box: int`, `empty: int`, `skipped_not_in_split: int`, `video_ids: tuple[str, ...]`.
  - `load_ground_truth(boxes_dir: Path, videos: pd.DataFrame, split: pd.DataFrame, role: str, ranges_path: Path | None = None) -> tuple[list[GroundTruthFrame], GroundTruthReport]`:
    - `role`이 `ROLES`에 없으면 `ValueError`.
    - 폴더의 `.txt` 파일마다 영상 id와 프레임을 읽는다. 숨김 파일(`.`으로 시작)은 건너뛴다. 영상이 `split`에 없으면 건너뛰고 `skipped_not_in_split`로 센다. 영상이 `split`에는 있는데 `videos`에 없으면 영상 id가 담긴 `ValueError`. 프레임이 `[0, frame_count)` 밖이면 영상 id와 프레임이 담긴 `ValueError`.
    - 그 영상의 `role`이 요청한 `role`일 때만 포함한다. 박스 파일이면 정규화 좌표를 `videos`의 `width`, `height`로 곱해 `(x1, y1, x2, y2)` 픽셀 좌표 한 개를 `boxes`에 담는다. 빈 파일이면 `boxes=()`(원판 없음 프레임, 평가에서 예측은 모두 오탐).
    - 결과는 `(video_id, frame_index)` 순(프레임은 숫자 순).
- 사람 단위 분할은 `split`의 `role`을 그대로 따른다. 같은 영상이 두 역할에 나오지 않는다. `person_id`로 사람 누수를 검사하는 것은 이 함수가 아니라 `split.assert_no_person_leak`이다.
- 구현에서 정한 규칙(컨트롤러 판단, 최종 리뷰 반영): 라벨 파일 이름의 프레임은 ASCII 숫자이고 앞에 0이 없어야 한다(`a_f03.txt`는 `PlateBoxStore`가 여는 `a_f3.txt`와 달라 박스가 조용히 빈 프레임이 되므로 `ValueError`). `.TXT`처럼 대소문자가 다른 확장자는 `ValueError`, 다른 확장자와 숨김 파일과 파일이 아닌 항목은 건너뛴다. `frame_count`, `width`, `height`는 정수여야 하며 아니면 영상 id와 열 이름이 담긴 `ValueError`다. `split`이나 `videos`에 필요한 열이 없으면 열 이름이 담긴 `ValueError`, 같은 `video_id`가 두 번 나와도 `ValueError`다. `ranges_path`가 있으면 범위 예외 검사는 역할 필터보다 먼저 모든 라벨 파일에 적용된다.
- `videos.csv`의 `width`, `height`, `frame_count`는 디코딩한 값과 같아야 한다(`video_timing.csv`와 54편 모두 일치함을 확인했다). 이 코드는 값을 다시 검사하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성** (합성 표와 임시 폴더에 쓴 합성 라벨 파일만 사용. 영상과 이미지는 쓰지 않는다).
  - `test_parse_label_name_reads_video_id_containing_f_and_frame_number`
  - `test_parse_label_name_rejects_other_names_with_the_name`
  - `test_box_file_becomes_pixel_box_from_normalized_values` (손으로 계산한 값, 예: 720×1280)
  - `test_empty_file_becomes_frame_without_boxes`
  - `test_only_requested_role_is_returned` / `test_unknown_role_raises_value_error`
  - `test_same_video_never_appears_in_both_roles`
  - `test_video_not_in_split_is_skipped_and_counted`
  - `test_video_in_split_but_not_in_videos_raises_value_error`
  - `test_frame_outside_video_raises_value_error`
  - `test_results_are_sorted_by_video_then_numeric_frame` (3번이 10번보다 앞)
  - `test_report_counts_boxes_and_empty_files`
  - `test_hidden_temp_files_are_ignored`
  - `test_loader_reads_and_writes_no_image_file` (폴더에 `.txt` 외에 새 파일이 없음)
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_ground_truth.py -v` → FAIL.
- [ ] **Step 3: 구현.** `plate_boxes.PlateBoxStore(boxes_dir, ranges_path)`로 읽는다(`status`, `read`).
- [ ] **Step 4: 통과 확인.** `uv run pytest -q` 전체 통과, `uv run ruff check`.
- [ ] **Step 5: 커밋.** 실제 데이터로는 PR #16의 라벨이 `main`에 들어온 뒤에 한 번 읽어 개수(학습 39, 검증 39, 빈 파일 6)를 확인한다. 이 Task에서는 하지 않는다.

**이 Task에서 제외:** 이미지 파일 생성(Task 6b), 모델 호출, 평가.

**후속(Task 5f 머지 뒤):** Task 5f(`plate_boxes.compare_passes`)도 라벨 파일 이름을 해석하는데 규칙이 다르다(앞의 0, 유니코드 숫자, `.DS_Store` 같은 숨김 파일 처리). 엄격한 `parse_label_name`과 파일 순회를 `plate_boxes.py` 한 곳에 두고 `ground_truth`, `compare_passes`, 첫 번째 라벨 잠금이 함께 쓰게 통합한다. 어느 pass를 정답으로 쓸지도 그때 팀이 정한다.

### Task 6b: 학습용 데이터셋 내보내기 (B단계 — fine-tuning 준비)

**상태:** 이미지 정책이 (a) 로컬 git 제외 폴더 저장 허용으로 정해졌고(이슈 #2, 2026-11-18, `docs/decisions/0003-baseline-setup.md`), 동료가 fine-tuning 순서를 확정했다(이슈 #2 13:05Z, `docs/yolo26-experiment-plan.md`). 첫 단계가 "학습 데이터셋 생성·점검"이다. 학습 구성은 학습 45장(원판 있음 39·없음 6, 8명·8편), 검증 39장(6명·6편)이다. 이 Task는 데이터셋 만들기와 점검까지이며 **학습은 시작하지 않는다.**

**배경:** YOLO 학습은 기본적으로 이미지 파일을 읽는다. `GroundTruthFrame` 목록(Task 6a)과 영상에서 프레임을 디코딩해 `out_dir`에 학습용 폴더를 만든다. `out_dir`은 git 제외 폴더여야 한다(`.gitignore`가 `data/` 아래 이미지 확장자를 이미 막는다). 얼굴이 나오는 이미지가 로컬 디스크에 생기며 저장소, 노트북 출력, PR 첨부에는 올리지 않는다.

**Files:**
- Create: `src/wl_snatch/dataset.py`, `scripts/build_plate_dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `GroundTruthFrame`(Task 6), `load_ground_truth`(Task 6a), `frames.read_frames`(Task 7).
- Produces (`dataset.py`):
  - `assert_ignored_output(out_dir: Path, repo_root: Path) -> None`: `out_dir`이 저장소 안이고 `git check-ignore`로 제외되지 않으면 `ValueError`(저장소 밖 경로는 허용).
  - `export_yolo_dataset(frames_by_role: Mapping[str, Sequence[GroundTruthFrame]], videos: pd.DataFrame, videos_root: Path, out_dir: Path, repo_root: Path, jpeg_quality: int = 95) -> Path`:
    - `out_dir`이 비어 있지 않으면 `ValueError`. 역할 키는 `train`, `validation`뿐이며 그 밖의 키는 `ValueError`.
    - 역할별로 `images/<role>/<video_id>_f<프레임>.jpg`와 `labels/<role>/<video_id>_f<프레임>.txt`(정규화 `0 cx cy w h`, 소수 6자리, 박스 없으면 빈 파일)를 쓰고, 상대경로 `dataset.yaml`(`train`, `val`, `names: {0: plate}`; `path` 키는 두지 않는다. Ultralytics가 상대 `path`를 학습 실행 위치 기준으로 풀기 때문에 키를 빼면 yaml이 있는 폴더가 기준이 된다)과 `manifest.json`을 쓴 뒤 `dataset.yaml` 경로를 돌려준다.
    - `manifest.json`: 역할별 이미지 수, 원판 있음·없음 프레임 수, 영상 id 목록, `jpeg_quality`, 이미지와 라벨 파일별 SHA256, 전체 SHA256(`<상대경로>:<파일 SHA256>` 줄을 경로순으로 이어 붙인 값의 해시. JPEG 바이트는 OpenCV 빌드에 따라 달라질 수 있어 같은 환경의 실행끼리만 비교한다).
    - 같은 `(video_id, frame)`이 두 역할에 있거나, `videos` 표의 같은 `person_id`가 두 역할에 모두 나오면 `ValueError`(사람 이름을 담는다). 프레임을 읽지 못하면 `ValueError`(영상 id와 프레임 번호를 담는다). 어떤 이유로든 실패하면 이 함수가 `out_dir` 아래에 만든 것을 지우고 오류를 다시 낸다.
    - 한 이미지의 크기는 `videos` 표의 `width`, `height`와 달라도 디코딩한 프레임의 실제 크기로 박스를 정규화한다(표와 다르면 `ValueError`).
  - `verify_exported_dataset(dataset_yaml: Path) -> DatasetReport`: 읽기 전용 점검. 모든 이미지에 같은 이름의 라벨이 있고 그 반대도 같다, 라벨 줄은 `0 cx cy w h`이고 값이 0 이상 1 이하이며 박스가 이미지 안에 있다, 파일의 SHA256이 `manifest.json`과 같다, `train`과 `val` 사이에 같은 파일 이름이 없다. 어긋나면 `ValueError`(어느 파일인지 담는다). `DatasetReport`는 frozen dataclass이며 역할별 이미지 수, 원판 있음·없음 수, 전체 해시를 가진다.
- `scripts/build_plate_dataset.py`: 인자 `--videos-csv`, `--split`, `--boxes-dir`, `--ranges`, `--videos-dir`, `--out`(기본 `data/derived/plate_dataset`). 두 역할을 `load_ground_truth`로 읽어 내보내고 `verify_exported_dataset`을 부른 뒤 역할별 개수와 전체 해시를 텍스트로 출력한다. 이미지를 화면이나 다른 곳으로 내보내지 않는다.
- 이 모듈 밖에서는 학습용 이미지 파일을 만들지 않는다.
- `validation` 역할은 `dataset.yaml`의 `val` 키로 쓰고 폴더 이름은 `validation`으로 둔다(역할에서 폴더와 키로의 대응을 코드에 명시한다).
- 이 Task에서 `data/README.md`가 학습용 이미지 생성을 허용하는 문장과 어긋나지 않는지 확인한다(이미 `data/derived/` 예외가 있다).

- [ ] **Step 1: 실패하는 테스트 작성** (합성 영상과 합성 `GroundTruthFrame`. 모델과 네트워크를 쓰지 않는다).
  - `test_export_places_frames_by_role_and_writes_yaml`
  - `test_yaml_uses_relative_paths_and_single_class_plate` (`path` 키 없음)
  - `test_label_text_is_normalized_from_pixel_box_and_empty_for_no_plate` (픽셀 박스는 손으로 계산한 값, 비정방 프레임으로 가로·세로 바뀜을 잡는다)
  - `test_empty_label_frames_are_exported_with_empty_label_files`
  - `test_export_rejects_non_empty_output_dir`
  - `test_assert_ignored_output_rejects_tracked_path_inside_repo` / `test_assert_ignored_output_allows_ignored_and_outside_paths`
  - `test_same_frame_in_two_roles_raises_value_error`
  - `test_same_person_in_both_roles_raises_value_error`
  - `test_unreadable_frame_raises_and_cleans_up_what_it_created`
  - `test_manifest_records_counts_and_sha256_of_every_file`
  - `test_verify_accepts_a_fresh_export_and_rejects_a_modified_label_or_image`
  - `test_verify_rejects_label_outside_the_image_and_orphan_files`
  - `test_export_writes_files_only_inside_out_dir`
  - `test_dataset_module_import_does_not_import_ultralytics`
- [ ] **Step 2~4:** 실패 확인, 구현, 통과 확인(전체 테스트와 `ruff`).
- [ ] **Step 5: 커밋.** 이미지, 영상, 가중치, 노트북 출력을 커밋하지 않는다.
- [ ] **Step 6: 실제 실행(Step 5 뒤, 컨트롤러).** `main`의 라벨로 `data/derived/plate_dataset/`에 만들어 학습 45(원판 있음 39·없음 6), 검증 39장과 전체 해시를 확인한다. 만든 이미지는 로컬에만 두고 어디에도 올리지 않는다.

**이 Task에서 제외:** 학습 실행, 증강, 이미지가 아닌 W&B·FiftyOne 연결, 데이터 보강, 이미지 시각화 파일 저장.

### Task 7: 탐지기와 설정 (A단계)

**Files:**
- Modify: `src/wl_snatch/frames.py`(`read_frames` 추가), `tests/test_frames.py`
- Create: `src/wl_snatch/config.py`, `src/wl_snatch/detector.py`, `scripts/run_detector.py`
- Test: `tests/test_config.py`, `tests/test_detector.py`

**Interfaces:**
- Consumes: `FrameDetections`, `BoxDetection`(Task 2), `frames.py`.
- Produces:
  - `frames.read_frames(video_path: Path, frame_indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]`: 영상을 한 번만 순차 디코딩해 요청한 프레임을 번호 오름차순으로 돌려준다(요청 순서와 무관, 중복 제거). 영상 길이 밖의 번호가 있으면 영상 이름과 번호가 담긴 `ValueError`. 이미지를 만들거나 저장하지 않는다.
  - `DetectorConfig(model_name: str, imgsz: int = 640, conf: float = 0.25, device: str = "auto", seed: int = 0, epochs: int = 50)` frozen dataclass. `imgsz`, `epochs`가 0 이하이거나 `conf`가 0~1 밖이면 `ValueError`.
  - `WEIGHT_NAMES = {"yolo26": "yolo26n.pt", "yolo11": "yolo11n.pt", "yolov8": "yolov8n.pt"}`와 `weight_name(model_name: str) -> str`(모르는 이름이면 목록이 담긴 `ValueError`). `yolo26n.pt`는 공식 문서에서 확인했고 나머지 둘은 Ultralytics 표준 이름이다(`docs/decisions/0001-environment.md`에 근거와 확인 한계를 적는다).
  - `resolve_device(requested: str, mps_available: bool, cuda_available: bool) -> tuple[str, str | None]` — 사용할 장치와, 요청과 다를 때의 사유. `"auto"`는 mps, cuda, cpu 순.
  - `ModelFn = Callable[[np.ndarray], Sequence[tuple[float, float, float, float, float]]]`: BGR 프레임 하나를 받아 `(x1, y1, x2, y2, confidence)`(픽셀) 목록을 돌려준다. 테스트는 가짜 `ModelFn`으로 한다.
  - `detect_frames(video_path: Path, video_id: str, frame_indices: Sequence[int], model: ModelFn, config: DetectorConfig) -> list[FrameDetections]`: 요청한 프레임마다 한 항목(번호 순). 신뢰도가 `config.conf` 미만인 박스는 뺀다. 영상을 열 수 없으면 요청한 프레임마다 `na_reason="video_unreadable"`, 영상 길이 밖의 번호는 `na_reason="frame_out_of_range"`(오류로 멈추지 않는다). 탐지가 0개인 프레임은 `na_reason=None`이고 박스가 없다.
  - `detect_video(video_path: Path, video_id: str, model: ModelFn, config: DetectorConfig) -> list[FrameDetections]`: 모든 프레임에 대한 `detect_frames`.
  - `load_ultralytics_model(config: DetectorConfig) -> ModelFn`: Ultralytics를 **함수 안에서** 가져와(`import ultralytics`는 모듈 맨 위에 두지 않는다) 가중치를 불러오고, 프레임을 `ModelFn` 형식으로 바꿔 준다. `resolve_device` 결과와 사유를 기록한다. 가중치 파일(`*.pt`)은 git 제외다.
- `pyproject.toml`의 의존성은 이 Task에서 바꾸지 않는다(`torch`, `torchvision`, `ultralytics`는 이미 기본 의존성). 선택 그룹으로 옮기는 일은 팀 합의 뒤 별도 변경으로 한다.

**구현에서 정한 규칙 (컨트롤러 판단, 리뷰 반영):**
- `frames.check_readable(video_path)`는 디코딩 없이 영상을 열 수 있는지만 확인하고 열 수 없으면 `VideoUnreadableError`(`ValueError`의 하위 클래스)를 낸다. `detect_frames`는 요청이 비어 있지 않으면 이것을 먼저 호출해, 열 수 없는 영상은 음수 번호를 포함한 모든 요청 프레임이 `video_unreadable`이 된다(빈 요청은 영상을 열지 않고 빈 목록).
- `frames.read_frames`는 도달할 수 있는 프레임을 모두 돌려준 뒤에야 범위 밖 번호에 대해 `ValueError`를 낸다. 음수 번호는 디코딩 전에 `ValueError`다. 제너레이터는 `try/finally`로 닫는다.
- `detect_video`는 영상을 열 수 없으면 `video_unreadable`인 프레임 0 하나를 돌려준다(프레임 수를 알 수 없기 때문).
- `DetectorConfig`는 빈 `model_name`, 불리언·소수·문자열 `imgsz`와 `epochs`, 불리언이나 숫자가 아닌 `conf`를 거부하고, `weights_path: Path | None = None`을 가진다. `resolve_device`는 `cuda:<숫자>`를 CUDA 요청으로 받는다.
- **사전학습 가중치로 `plate`를 쓰지 않는다 (최종 리뷰 B2):** `WEIGHT_NAMES`의 가중치는 COCO 사전학습이라 원판 탐지기가 아니다. `load_ultralytics_model`은 `weights_path`가 없으면 Ultralytics를 가져오기 전에 `ValueError`를 내고, 불러온 모델의 `names`가 `{0: "plate"}`가 아니면 `check_plate_names`가 `ValueError`를 낸다. `weight_name()`은 학습의 시작점(Task 8)에만 쓴다. `scripts/run_detector.py`는 `--weights PATH`가 필요하다.

- [ ] **Step 1: 실패하는 테스트 작성** (합성 영상과 가짜 `ModelFn`만 사용. 가중치를 받거나 Ultralytics를 호출하지 않는다).
  - `test_read_frames_returns_requested_frames_in_ascending_order_once_each` (`tests/test_frames.py`)
  - `test_read_frames_out_of_range_raises_value_error_with_name_and_index` (`tests/test_frames.py`)
  - `test_config_rejects_nonpositive_imgsz_and_epochs`
  - `test_config_rejects_conf_outside_zero_one`
  - `test_weight_name_maps_known_models_and_rejects_unknown`
  - `test_resolve_device_returns_requested_when_available`
  - `test_resolve_device_unavailable_returns_cpu_with_reason`
  - `test_auto_prefers_mps_then_cuda_then_cpu`
  - `test_detect_frames_calls_injected_model_once_per_requested_frame`
  - `test_detect_frames_drops_boxes_below_conf_and_keeps_zero_detection_frames_as_not_na`
  - `test_detect_frames_marks_out_of_range_frame_as_na_and_continues`
  - `test_unreadable_video_gives_na_for_every_requested_frame`
  - `test_detect_video_covers_every_frame`
  - `test_module_import_does_not_import_ultralytics` (모듈을 가져와도 `ultralytics`가 `sys.modules`에 없음)
  - `test_detector_writes_no_image_file`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_frames.py tests/test_config.py tests/test_detector.py -v` → 새 테스트 FAIL.
- [ ] **Step 3: 구현.** 가짜 모델로 테스트가 통과하게 한 뒤 `load_ultralytics_model`을 붙인다. `load_ultralytics_model`은 테스트하지 않고(가중치와 네트워크가 필요), `scripts/run_detector.py`가 영상 하나에 돌려 `FrameDetections`를 CSV로 쓰는 실행기로 쓴다(이 Task에서는 실행하지 않는다).
- [ ] **Step 4: 통과 확인.** `uv run pytest -q` 전체 통과, `uv run ruff check`.
- [ ] **Step 5: 커밋.**

**이 Task에서 제외:** 학습(Task 8), 가중치 내려받기와 실제 모델 실행, 이미지 파일 생성, 의존성 변경.

### Task 7b: 층 사이 통합 테스트 (A단계)

**배경:** 층마다 따로 시험하면 층 사이의 약속(좌표 규칙, `N/A`, 프레임 번호)이 어긋나도 알 수 없다. 합성 영상, 합성 라벨, 가짜 모델로 정답 만들기 → 탐지 → 평가를 끝에서 끝까지 확인한다.

**Files:** Create `tests/test_pipeline.py`.

- 가짜 모델의 박스는 정답 만들기의 출력이 아니라 **손으로 계산한 픽셀 상수**로 정한다. 정답 코드의 출력을 그대로 돌려주면 변환 오류가 상쇄되어 테스트가 틀린 구현도 통과시킨다(컨트롤러가 정답 만들기에서 너비와 높이를 바꿔 이를 확인했다).
- 상수는 좁고 한쪽으로 치우친 박스여야 너비·높이 혼동이 IoU를 바꾼다.
- 검증 방법: 변형 두 가지를 임시로 적용해 테스트가 실패하는지 본다. 정답 만들기가 너비 자리에 높이를 읽는 변형(통합 테스트 9개 중 7개 실패), 탐지기의 신뢰도 필터를 뒤집은 변형(9개 중 6개 실패). 확인한 뒤 되돌린다.
- 테스트: `test_perfect_model_gives_precision_and_recall_one_across_the_layers`, `test_box_on_a_labeled_empty_frame_is_a_false_positive_across_the_layers`, `test_shifted_box_below_threshold_is_a_miss_and_a_false_positive_across_the_layers`, `test_unreadable_video_is_counted_as_na_by_reason_across_the_layers`, `test_frame_beyond_the_video_is_na_frame_out_of_range_across_the_layers`, `test_only_the_requested_role_flows_through`, `test_ground_truth_pixels_match_hand_computed_boxes_across_the_layers`, `test_pipeline_writes_no_image_file`, `test_detections_survive_the_csv_boundary_between_layers`.

### Task 7c: 정답으로 평가하기 `evaluation_run.py` (A단계)

**배경:** "정답을 영상별로 묶어 탐지하고 평가한다"는 반복은 스크립트에 두지 않고 `src/`에 둔다(스크립트는 호출만 한다). 베이스라인 비교(Task 8)가 이것을 쓴다.

**Files:** Create `src/wl_snatch/evaluation_run.py`, `tests/test_evaluation_run.py`. 통합 테스트(`tests/test_pipeline.py`)의 평가 반복도 이 함수를 쓴다.

- `evaluate_ground_truth(truths: Sequence[GroundTruthFrame], video_paths: Mapping[str, Path], model: ModelFn, config: DetectorConfig, iou_threshold: float) -> DetectionMetrics`: `truths`를 `video_id`로 묶어(오름차순) 영상마다 `detect_frames`를 한 번 부르고 `evaluate_detections`로 평가한다. `video_paths`에 없는 `video_id`는 `ValueError`. 빈 `truths`는 모든 비율이 `None`이고 `frames_evaluated == 0`이다.

### Task 8a: COCO 사전학습 모델을 그대로 실행하는 탐색 (A2단계, 이슈 #2)

**상태:** 도메인 담당이 2026-11-18 #2에서 요청을 정정했다. 원판 데이터로 fine-tuning하기 전에, COCO 사전학습 모델을 그대로 실행해 로드와 실행 가능 여부, 원래 클래스로 탐지되는 물체, 추론 시간을 확인한다(`docs/decisions/0003-baseline-setup.md`). 이 결과는 **원판 탐지 성능이 아니다.** 원판 라벨과의 IoU, 정밀도, 재현율은 계산하지 않고, 다른 클래스의 박스를 `plate`로 바꾸지 않는다.

**Files:**
- Modify: `src/wl_snatch/detector.py`(클래스 이름을 함께 돌려주는 일반형 추가), `tests/test_detector.py`
- Create: `src/wl_snatch/coco_baseline.py`, `scripts/run_coco_baseline.py`
- Test: `tests/test_coco_baseline.py`

**Interfaces:**
- Consumes: `DetectorConfig`, `resolve_device`, `weight_name`, `FrameDetections`, `BoxDetection`, `ground_truth.load_ground_truth`(대상 프레임 번호만 쓰고 박스는 쓰지 않는다), `frames.read_frames`.
- Produces:
  - `detector.LabeledModelFn = Callable[[np.ndarray], Sequence[tuple[float, float, float, float, float, str]]]`: BGR 프레임 하나를 받아 `(x1, y1, x2, y2, confidence, label)`(픽셀) 목록을 돌려준다.
  - `detector.detect_frames_labeled(video_path, video_id, frame_indices, model: LabeledModelFn, config) -> list[FrameDetections]`: `detect_frames`와 같은 규칙(오름차순, 중복 제거, 신뢰도 미만 제거, `video_unreadable`, `frame_out_of_range`)으로 `BoxDetection.label`에 모델이 준 이름을 쓴다. 기존 `detect_frames`는 이것을 불러 라벨을 `"plate"`로 고정하는 얇은 층으로 바꾼다(기존 동작과 테스트는 그대로).
  - `coco_baseline.load_coco_model(config: DetectorConfig) -> LabeledModelFn`: Ultralytics를 함수 안에서만 가져와 `weight_name(config.model_name)`의 COCO 사전학습 가중치를 불러오고, 원래 클래스 이름을 유지해 돌려준다. `config.weights_path`가 있으면 `ValueError`(학습한 가중치는 `plate` 경로로). 불러온 모델의 `names`가 80개 COCO 클래스가 아니거나 `plate`를 포함하면 `ValueError`(이 모드는 COCO 사전학습 모델만 받는다). 장치는 `resolve_device`로 정하고 사유를 기록한다.
  - `coco_baseline.resolve_config_device(config) -> tuple[str, str | None]`: torch를 함수 안에서 가져와 장치를 정한다. `load_coco_model`과 스크립트의 환경 기록이 이 함수 하나를 쓴다(모델이 쓴 장치와 기록한 장치가 같다).
  - `coco_baseline.check_coco_names(names)`(순수 함수): 키가 0..79인 80개 항목이고 `plate`라는 이름이 없어야 한다. 아니면 처음 5개 이름이 담긴 `ValueError`.
  - `coco_baseline.environment_record(config, device, reason, versions: Mapping[str, str] | None = None) -> dict[str, str]`: Python, `ultralytics`, `torch`, `opencv`의 버전(`importlib.metadata`, 테스트에서는 `versions`로 주입), 운영체제와 기계 정보(`platform`), 요청한 장치와 선택된 장치와 사유, 가중치 이름, `imgsz`, `conf`, NMS의 IoU 기본값(알 수 있을 때), `seed`를 돌려준다.
  - `coco_baseline.timed(model: LabeledModelFn) -> tuple[LabeledModelFn, list[float]]`: 호출마다 걸린 시간(ms)을 두 번째 값에 쌓는 감싸개.
  - `CocoRun(summaries: dict[str, CocoSummary], warmup_ms: float | None)`와 `run_coco_baseline(truths_by_role, video_paths, model, config) -> CocoRun`: 역할별로 라벨된 프레임에서 모델을 부른다. 시간 측정 전에 첫 프레임에서 모델을 **한 번 따로 호출(웜업)**하고, 그 시간은 `warmup_ms`로 한 번만 보고하며 어떤 역할의 시간 통계와 클래스 개수에도 넣지 않는다(읽을 수 있는 프레임이 없으면 `None`).
  - `CocoSummary` frozen dataclass와 `summarize_coco_run(frames: Sequence[FrameDetections], times_ms: Sequence[float]) -> CocoSummary`: 클래스 이름별 탐지 개수, 박스가 하나도 없는 처리된 프레임 수, 처리한 프레임 수, `N/A` 프레임 수와 사유별 개수, 첫 호출 시간, 나머지의 중앙값, 평균, 최댓값(호출이 없으면 `None`).
  - `coco_baseline.format_report(record: Mapping[str, str], summaries: Mapping[str, CocoSummary], warmup_ms: float | None = None) -> str`: 설정과 역할별 수치만 담은 텍스트(이미지, 좌표 없음). 원판 성능을 나타내는 문구를 만들지 않는다.
- `scripts/run_coco_baseline.py`: 인자 `--model`(`yolo26`, `yolo11`, `yolov8`), `--out`(텍스트 보고서 경로, 기본 `artifacts/coco_baseline/<model>.txt`, git 제외 폴더). 라벨된 프레임을 역할별로 `load_ground_truth`로 얻어 영상별로 `detect_frames_labeled`를 부르고, 보고서를 쓴다. 평가기를 부르지 않는다. 이 Task에서는 실행하지 않는다(가중치를 받는 것은 별도 승인 뒤).

- [ ] **Step 1: 실패하는 테스트 작성** (합성 영상과 가짜 `LabeledModelFn`만 사용. 가중치를 받거나 Ultralytics를 부르지 않는다).
  - `test_detect_frames_labeled_keeps_model_labels`
  - `test_detect_frames_labeled_applies_the_same_na_and_order_rules_as_detect_frames`
  - `test_detect_frames_still_labels_every_box_plate_after_the_refactor` (`tests/test_detector.py`의 기존 테스트가 그대로 통과)
  - `test_load_coco_model_refuses_a_weights_path_before_importing_ultralytics` (새 인터프리터에서 `ultralytics`가 `sys.modules`에 없음)
  - `test_check_coco_names_accepts_80_classes_and_rejects_plate_or_other_sizes` (순수 함수로 분리해 가짜 `names`로 시험)
  - `test_environment_record_contains_versions_device_and_conditions` (`versions` 주입)
  - `test_timed_records_one_time_per_call_in_order`
  - `test_summarize_counts_classes_zero_box_frames_and_na_reasons`
  - `test_summarize_time_stats_separate_the_first_call_and_handle_no_calls`
  - `test_report_contains_settings_and_counts_and_no_plate_performance_wording`
  - `test_coco_module_import_does_not_import_ultralytics`
  - `test_coco_baseline_writes_no_image_file`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_coco_baseline.py tests/test_detector.py -v` → 새 테스트 FAIL.
- [ ] **Step 3: 구현.** `load_coco_model`은 가중치와 네트워크가 필요하므로 테스트하지 않는다.
- [ ] **Step 4: 통과 확인.** `uv run pytest -q` 전체 통과, `uv run ruff check`.
- [ ] **Step 5: 커밋.** 가중치를 받거나 모델을 실행하지 않는다.
- [ ] **Step 6: 실제 실행(별도 승인 뒤).** 가중치를 내려받아 세 모델을 같은 조건(입력 640, `conf` 0.25, 같은 장치)에서 라벨된 84프레임(학습 45, 검증 39)에 실행하고 보고서를 이슈 #2에 수치와 텍스트로 올린다. 입력 색상 순서와 좌표를 확인한다: `result.orig_shape`가 프레임 크기와 같은지, 박스가 프레임 안에 있는지, 같은 프레임을 BGR과 RGB로 넣었을 때 결과 차이(참고용).

**이 Task에서 제외:** 원판 라벨 기준 평가, fine-tuning, 이미지 파일 생성, 가중치 내려받기.

### Task 8b: 탐지 결과를 로컬 화면에서 확인하는 뷰어 (A2단계, 이슈 #2)

**상태:** Task 8a의 COCO 실행 결과에서 `frisbee`, `clock` 같은 박스가 무엇을 감쌌는지는 확인하지 못했다(PR #18, 이슈 #2). 박스를 프레임 위에 **메모리에서만** 그려 로컬 화면에서 눈으로 확인하는 뷰어를 만든다. 얼굴이 담긴 화면이므로 이미지 파일을 쓰지 않고, 스크린샷을 이슈와 PR에 올리지 않는다. 이 Task는 성능을 계산하지 않는다.

**Files:**
- Create: `src/wl_snatch/detection_viewer.py`, `tutorials/view_detections.ipynb`
- Test: `tests/test_detection_viewer.py`

**Interfaces:**
- Consumes: `FrameDetections`, `BoxDetection`, `GroundTruthFrame`, `frames.read_frames`, `detector.detect_frames_labeled`, `coco_baseline.load_coco_model`, `ground_truth.load_ground_truth`.
- Produces:
  - `class_color(label: str) -> tuple[int, int, int]`: 라벨 이름에서 `hashlib.md5`로 정한 BGR 색(실행마다 같다).
  - `draw_detections(frame, boxes: Sequence[BoxDetection], *, min_confidence=0.0, label_filter: Collection[str] | None = None, color_for=class_color) -> np.ndarray`: 입력을 바꾸지 않고 복사본에 상자, 라벨, 신뢰도를 그린다. 프레임 밖으로 나간 좌표는 잘라서 그린다.
  - `draw_truth(frame, boxes: Sequence[BoxDetection]) -> np.ndarray`: 탐지 색과 다른 단색 상자(참고용 원판 라벨).
  - `DetectionViewSession(video_path, video_id, detections: Sequence[FrameDetections], truths: Sequence[GroundTruthFrame] = (), roles: Mapping[int, str] | None = None)`: `detections`에 있는 프레임만 보여 주고 한 번만 읽는다(`read_frames`). `frame_indices`, `goto(i)`, `step(delta)`(양끝에서 멈춤), `classes()`, `view(show_truth=False, label_filter=None, min_confidence=0.0) -> tuple[np.ndarray, list[str]]`(그린 프레임과 정보 줄: 프레임 번호, 역할, 클래스별 개수). `N/A` 프레임은 사유를 정보 줄에 보여 준다. `show()`는 ipywidgets를 메서드 안에서만 가져와 화면을 만든다(슬라이더, 클래스 선택, 신뢰도 하한, "참고용 원판 라벨" 체크상자 기본 꺼짐).
  - `build_view_session(config, model, video_id, videos_dir, boxes_dir, videos, split, ranges_path=None) -> DetectionViewSession`: `load_ground_truth`로 그 영상의 라벨된 프레임만 얻어 그 프레임에서만 `model`을 실행한다.
- `tutorials/view_detections.ipynb`: 첫 셀에 화면 캡처를 이슈와 PR에 올리지 말라는 경고, 설정 셀(모델 이름, 영상 ID), 세션 셀, `session.show()`. 코드는 패키지를 부르기만 한다.

**규칙:** 원판 라벨 겹침은 "참고용 원판 라벨"로 표시하고 IoU나 정밀도 같은 수치를 계산하거나 보여 주지 않는다. 이미지 파일을 쓰지 않는다. 한 뷰어는 한 모델의 결과만 보여 준다.

- [ ] **Step 1: 실패하는 테스트 작성** (합성 영상과 가짜 모델만 사용)
  - `test_draw_detections_returns_a_copy_and_keeps_the_input`
  - `test_class_color_is_stable_and_differs_between_labels`
  - `test_draw_detections_clips_boxes_outside_the_frame`
  - `test_session_shows_only_frames_present_in_detections`
  - `test_goto_and_step_clamp_at_both_ends`
  - `test_view_filters_by_label_and_min_confidence`
  - `test_truth_overlay_is_off_by_default_and_uses_another_color`
  - `test_info_lines_contain_frame_role_and_class_counts`
  - `test_na_frame_is_shown_with_its_reason`
  - `test_viewer_writes_no_image_file`
  - `test_module_import_does_not_import_ultralytics_or_ipywidgets` (새 인터프리터)
  - `test_build_view_session_runs_the_model_only_on_labeled_frames_of_that_video`
- [ ] **Step 2: 실패 확인.** `uv run pytest tests/test_detection_viewer.py -v`
- [ ] **Step 3: 구현.** 그리기는 `cv2`, 화면은 `ipywidgets`와 메모리 안의 `cv2.imencode`만 쓴다.
- [ ] **Step 4: 통과 확인.** `uv run pytest -q`, `uv run ruff check`, 노트북 위생 검사.
- [ ] **Step 5: 커밋.**

**이 Task에서 제외:** 이미지 파일 저장, 성능 지표, 모델 비교, fine-tuning.

### Task 8: 모델 비교와 결정 기록

**상태:** Task 8a(COCO 사전학습 그대로 실행)의 결과를 함께 본 뒤 fine-tuning 여부와 첫 학습 설정(backbone 동결, epoch, 학습률, 배치, seed)을 정한다(`docs/decisions/0003-baseline-setup.md`). 그 전에는 학습을 시작하지 않는다. 학습 구성은 학습 45장(원판 있음 39·없음 6), 검증 39장이다. 빈 라벨 6장은 학습용이고, 원판 없음 독립 평가는 별도 인원의 라벨이 생길 때까지 미평가로 명시한다.

**Files:**
- Create: `src/wl_snatch/compare.py`, `src/wl_snatch/train.py`, `scripts/train_detector.py`, `scripts/compare_detectors.py`, `docs/decisions/0004-detector-baseline.md`
- Test: `tests/test_compare.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `DetectorConfig`, `evaluate_ground_truth`(Task 7c), `DetectionMetrics`, `weight_name`(학습의 시작점).
- Produces:
  - `train_detector(config: DetectorConfig, dataset_yaml: Path, project_dir: Path) -> Path`(`src/wl_snatch/train.py`): Ultralytics를 함수 안에서 가져와 `weight_name(config.model_name)`에서 시작해 단일 클래스 `plate`로 학습하고, 학습된 가중치 파일(`best.pt`)의 경로를 돌려준다. `project_dir`은 git 제외 폴더(`runs/` 아래)여야 하고, 그 경로의 검사는 `dataset.assert_ignored_output`을 쓴다. 학습 뒤 평가는 그 경로를 `DetectorConfig.weights_path`로 주어 `load_ultralytics_model`로 불러온다. `scripts/train_detector.py`는 호출만 한다.
  - `ComparisonRow(model_name: str, seed: int, imgsz: int, conf: float, iou_threshold: float, metrics: DetectionMetrics)`
  - `write_comparison_csv(rows: list[ComparisonRow], path: Path) -> None`
  - `assert_single_axis(rows: list[ComparisonRow]) -> None` — `model_name`을 제외한 조건(seed, imgsz, conf, iou_threshold)이 모든 행에서 같지 않으면 `ValueError`. 또한 모든 행의 `metrics.frames_evaluated`와 `metrics.na_reasons`가 같아야 하고(같은 프레임에서 비교), `frames_evaluated`가 0인 행이 있으면 `ValueError`다(평가된 프레임이 없는 비교는 `N/A`와 사유로 남긴다).

- [ ] **Step 1: 실패하는 테스트 작성.**
  - `test_rows_with_different_seed_are_rejected`
  - `test_rows_differing_only_by_model_are_accepted`
  - `test_rows_evaluated_on_different_frames_are_rejected`
  - `test_row_with_zero_evaluated_frames_is_rejected`
  - `test_csv_contains_one_row_per_model_with_metrics`
  - `test_csv_writes_na_counts_and_none_metrics_as_empty`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_compare.py -v` → FAIL.
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.** → 4 passed.
- [ ] **Step 5: YOLO26 기준선 실행(B단계, 이슈 #2의 이미지 정책과 학습 구성 답변 뒤).** `scripts/train_detector.py`(`train_detector`)로 YOLO26을 Task 6b가 만든 학습용 폴더의 train으로 학습하고, 검증 영상의 라벨된 프레임에서 `load_ground_truth`(Task 6a)와 `evaluate_ground_truth`(Task 7c, `detect_frames`로 메모리 디코딩)로 평가한다. M2 8GB에서 메모리 때문에 실패하면 `imgsz`나 배치를 줄이고 그 사실을 기록한다.
- [ ] **Step 6: YOLO11, YOLOv8 비교 실행.** 모델만 바꾸고 나머지는 같게 한다. `scripts/compare_detectors.py`가 `assert_single_axis`를 먼저 호출한다.
- [ ] **Step 7: 결정 기록.** `docs/decisions/0004-detector-baseline.md`에 비교 표, 조건, seed, 실패 사례(오탐 중심), `N/A` 개수를 적는다. 비교 표에는 학습·검증 영상 수와 프레임 수(N), 프레임 선정 규칙(단계 경계 + 사이 프레임), 음성 프레임 수, 그룹을 함께 적는다. 해석은 측정값 범위에서만 쓴다.
- [ ] **Step 8: 커밋.**

### Task 9: 노트북 위생 검사와 입문 튜토리얼

**Files:**
- Create: `tutorials/01_detection_quickstart.py`, `tutorials/01_detection_quickstart.ipynb`
- 노트북 위생 검사(`tests/test_notebook_hygiene.py`)는 Task 5c에서 만들었다. 이 Task는 새 노트북이 그 검사를 통과하는지만 확인한다.

**Interfaces:**
- Consumes: `detect_video`, `load_ultralytics_model`, `DetectorConfig`, `read_detections_csv`.

- [ ] **Step 1: 위생 검사 확인.** `tests/test_notebook_hygiene.py`(Task 5c에서 작성)가 `tutorials/`와 저장소의 모든 `.ipynb`에 적용된다: 코드 셀에 `def`, `class`, `lambda` 없음, `assert`, `unittest`, `pytest` 문자열 없음, 한 셀 15줄 이하, 저장된 출력 없음, `tests/`가 노트북을 실행하는 라이브러리를 가져오지 않음.
- [ ] **Step 2:** (Task 5c로 이동. 이 Task에서는 건너뛴다.)
- [ ] **Step 3: 튜토리얼 작성.** `.py`는 "설정 → 영상 하나에 탐지 → 박스 그리기 → 결과 읽기" 순서로 `src/`를 불러 쓰기만 한다. 노트북은 같은 흐름을 셀로 나눈 얇은 버전이고, 설명은 마크다운 셀에 적는다.
- [ ] **Step 4: 통과 확인.** Run: `uv run pytest tests/test_notebook_hygiene.py -v` → 새 노트북을 포함해 통과.
- [ ] **Step 5: 전체 실행.** Run: `uv run pytest -v` → 전부 통과. Run: `uv run python tutorials/01_detection_quickstart.py` → 영상 하나의 탐지 결과를 읽어 개수와 좌표를 출력한다. 이미지를 파일로 저장하지 않는다(#11). 박스 확인은 Task 5b의 `render_label_frame`이 돌려준 배열을 화면에 표시한다.
- [ ] **Step 6: 커밋.**
