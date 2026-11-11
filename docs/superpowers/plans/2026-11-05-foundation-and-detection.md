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

### Task 6: 탐지 평가기

**Files:**
- Create: `src/wl_snatch/eval_detection.py`
- Test: `tests/test_eval_detection.py`

**Interfaces:**
- Consumes: `FrameDetections`, `BoxDetection` (Task 2).
- Produces:
  - `GroundTruthFrame(video_id: str, frame_index: int, boxes: tuple[tuple[float, float, float, float], ...])` frozen dataclass.
  - `DetectionMetrics(precision: float | None, recall: float | None, false_positives_per_frame: float | None, frames_evaluated: int, frames_na: int, na_reasons: dict[str, int])`
  - `evaluate_detections(predictions: list[FrameDetections], truths: list[GroundTruthFrame], iou_threshold: float) -> DetectionMetrics`

**동작:** 오탐은 정답과 IoU가 임계값 미만이거나 이미 매칭된 정답에 또 매칭된 예측이다. 정답 박스가 없는 프레임(원판 없음)의 예측은 모두 오탐이다. `N/A` 프레임은 평균에서 빼고 사유별로 센다. 분모가 0이면 해당 지표는 `None`이다.

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
- [ ] **Step 4: 통과 확인.** → 7 passed.
- [ ] **Step 5: 커밋.**

### Task 6b: 분할 기준 데이터셋과 정답 로더 (사전 점검에서 추가)

**Files:**
- Create: `src/wl_snatch/dataset.py`, `scripts/build_plate_dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `make_person_split` 결과(Task 4), `GroundTruthFrame`(Task 6).
- Produces:
  - `parse_frame_name(name: str) -> tuple[str, int]` — `c02_sn_f_01_frame_0109.jpg` → `("c02_sn_f_01", 109)`. 형식이 다르면 `ValueError`.
  - `build_yolo_dataset(image_dirs: list[Path], label_dirs: list[Path], split: pd.DataFrame, out_dir: Path) -> Path` — 라벨된 프레임을 새 사람 단위 분할(`role`)에 따라 `out_dir/images|labels/{train,validation}`로 복사하고, 상대경로 `dataset.yaml`(`names: {0: plate}`)을 쓴 뒤 그 경로를 돌려준다. 분할에 없는 영상의 프레임은 복사하지 않고 개수를 반환 로그에 남긴다. 같은 파일명이 두 폴더에 있으면 `ValueError`.
  - `load_ground_truth(images_dir: Path, labels_dir: Path) -> list[GroundTruthFrame]` — YOLO 정규화 좌표를 이미지 크기로 곱해 픽셀 좌표로 바꾼다.

**이유:** 아카이브의 `images/train|validation`은 옛 분할이다. 새 사람 단위 분할과 섞이면 같은 사람이 양쪽에 들어가 평가가 무효가 된다. 평가는 정답 박스가 있는 라벨된 프레임에서만 한다. 원판 없는 프레임(음성)이 데이터에 없으면 그 개수 0을 보고서에 명시하고, 오탐은 라벨된 프레임의 정답 밖 박스로만 센다.

- [ ] **Step 1: 실패하는 테스트 작성.**
  - `test_parse_frame_name_extracts_video_id_and_index`
  - `test_parse_frame_name_rejects_unexpected_format`
  - `test_build_dataset_places_frames_by_person_role`
  - `test_build_dataset_writes_relative_dataset_yaml`
  - `test_build_dataset_skips_frames_of_unsplit_videos_and_reports_count`
  - `test_build_dataset_rejects_duplicate_filename_across_dirs`
  - `test_no_person_appears_in_both_train_and_validation_folders`
  - `test_load_ground_truth_converts_normalized_to_pixels`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_dataset.py -v` → FAIL.
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.** → 8 passed.
- [ ] **Step 5: 실제 실행.** `plate_dataset`과 `plate_dataset_v2`의 이미지·라벨을 합쳐 `data/plate_dataset_split/`를 만든다. 사람 누수 검사를 통과해야 한다. (Checkpoint에서 재라벨링이 정해졌다면 그 라벨을 쓴다.)
- [ ] **Step 6: 커밋.**

### Task 7: 탐지기와 설정

**Files:**
- Create: `src/wl_snatch/config.py`, `src/wl_snatch/detector.py`, `scripts/train_detector.py`, `scripts/run_detector.py`
- Test: `tests/test_config.py`, `tests/test_detector.py`

**Interfaces:**
- Produces:
  - `DetectorConfig(model_name: str, imgsz: int = 640, conf: float = 0.25, device: str = "auto", seed: int = 0, epochs: int = 50)` frozen dataclass. `imgsz`, `epochs`가 0 이하이거나 `conf`가 0~1 밖이면 `ValueError`.
  - `resolve_device(requested: str, mps_available: bool, cuda_available: bool) -> tuple[str, str | None]` — 사용할 장치와, 요청과 다를 때의 사유.
  - `detect_video(video_path: Path, video_id: str, model: Callable, config: DetectorConfig) -> list[FrameDetections]` — `model`은 프레임을 받아 박스 목록을 돌려주는 호출 가능 객체이므로 테스트에서 가짜로 바꿀 수 있다. 읽을 수 없는 영상은 `na_reason="video_unreadable"`인 프레임 하나를 돌려준다.
  - `detect_images(image_paths: list[Path], model: Callable, config: DetectorConfig) -> list[FrameDetections]` — 파일명을 `parse_frame_name`(Task 6b)으로 해석해 `video_id`, `frame_index`를 채운다. 읽을 수 없는 이미지는 `na_reason="image_unreadable"`. (평가는 영상이 아니라 라벨된 프레임 이미지에서 한다.)
  - `load_ultralytics_model(config: DetectorConfig) -> Callable` — Ultralytics를 감싼다. 가중치 이름은 Step 1에서 확인한 값을 쓴다.

- [ ] **Step 1: 가중치 이름 확인.** 공식 문서로 YOLO26, YOLO11, YOLOv8 nano 계열의 정확한 가중치 이름을 확인해 `docs/decisions/0001-environment.md`에 추가한다. 학습은 `dataset.yaml`의 `plate` 단일 클래스로 파인튜닝한다.
- [ ] **Step 2: 실패하는 테스트 작성.**
  - `test_config_rejects_nonpositive_imgsz_and_epochs`
  - `test_config_rejects_conf_outside_zero_one`
  - `test_resolve_device_returns_requested_when_available`
  - `test_resolve_device_unavailable_returns_cpu_with_reason`
  - `test_auto_prefers_mps_then_cuda_then_cpu`
  - `test_detect_video_uses_injected_model_for_every_frame`
  - `test_detect_images_fills_video_id_and_frame_index_from_filename`
  - `test_unreadable_image_gives_na_frame_with_reason`
  - `test_unreadable_video_gives_na_frame_with_reason`
- [ ] **Step 3: 실패 확인.** Run: `uv run pytest tests/test_config.py tests/test_detector.py -v` → FAIL.
- [ ] **Step 4: 구현.** 가짜 모델로 테스트가 통과하게 한 뒤 `load_ultralytics_model`을 붙인다.
- [ ] **Step 5: 통과 확인.** → 9 passed.
- [ ] **Step 6: 커밋.**

### Task 8: 모델 비교와 결정 기록

**Files:**
- Create: `src/wl_snatch/compare.py`, `scripts/compare_detectors.py`, `docs/decisions/0003-detector-baseline.md`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `DetectorConfig`, `evaluate_detections`, `DetectionMetrics`.
- Produces:
  - `ComparisonRow(model_name: str, seed: int, imgsz: int, conf: float, iou_threshold: float, metrics: DetectionMetrics)`
  - `write_comparison_csv(rows: list[ComparisonRow], path: Path) -> None`
  - `assert_single_axis(rows: list[ComparisonRow]) -> None` — `model_name`을 제외한 조건(seed, imgsz, conf, iou_threshold)이 모든 행에서 같지 않으면 `ValueError`.

- [ ] **Step 1: 실패하는 테스트 작성.**
  - `test_rows_with_different_seed_are_rejected`
  - `test_rows_differing_only_by_model_are_accepted`
  - `test_csv_contains_one_row_per_model_with_metrics`
  - `test_csv_writes_na_counts_and_none_metrics_as_empty`
- [ ] **Step 2: 실패 확인.** Run: `uv run pytest tests/test_compare.py -v` → FAIL.
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.** → 4 passed.
- [ ] **Step 5: YOLO26 기준선 실행.** `scripts/train_detector.py`로 YOLO26을 `data/plate_dataset_split/`(Task 6b)의 train으로 학습하고, validation 이미지에서 `detect_images`와 `load_ground_truth`, 평가기를 돌린다. M2 8GB에서 메모리 때문에 실패하면 `imgsz`나 배치를 줄이고 그 사실을 기록한다.
- [ ] **Step 6: YOLO11, YOLOv8 비교 실행.** 모델만 바꾸고 나머지는 같게 한다. `scripts/compare_detectors.py`가 `assert_single_axis`를 먼저 호출한다.
- [ ] **Step 7: 결정 기록.** `docs/decisions/0003-detector-baseline.md`에 비교 표, 조건, seed, 실패 사례(오탐 중심), `N/A` 개수를 적는다. 해석은 측정값 범위에서만 쓴다.
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
