# 0001. 실행 환경과 버전

확인일: 2026-11-05. 상태: 초안 (팀 리뷰 전).

## 결정

- Python 3.12 (`.python-version`, `requires-python = ">=3.12"`). spec 9절의 기본값을 따랐다.
- 패키지 관리: uv. 잠금 파일 `uv.lock`을 커밋한다.
- 의존성: matplotlib, numpy, opencv-python, pandas, torch, torchvision, ultralytics. dev 그룹: pytest, jupyterlab. 버전 상한은 두지 않고 `uv.lock`으로 고정한다.

## 확인한 내용

| 항목 | 확인 결과 | 출처 |
|---|---|---|
| Python 지원 버전 | 3.11~3.15가 지원 상태. 3.12는 보안 수정 단계, 종료 예정 2028-10. 3.15는 2026-10-09 출시 | https://devguide.python.org/versions/ |
| Ultralytics 최신 | PyPI 최신 8.4.202. `requires_python >=3.8`, `torch>=1.8.0`, `torchvision>=0.9.0` | https://pypi.org/pypi/ultralytics/json |
| PyTorch 최신 | PyPI 최신 2.15.0. `requires_python >=3.11` | https://pypi.org/pypi/torch/json |
| YOLO26 | Ultralytics 모델로 존재(2026년 1월 출시로 표기). 필요한 최소 ultralytics 버전은 문서에 명시되지 않음 | https://docs.ultralytics.com/models/yolo26/ |
| 실제 해결된 조합 | `uv sync` 결과 Python 3.12.7, torch 2.15.0, torchvision 0.29.2, ultralytics 8.4.202. 이 Mac에서 `torch.backends.mps.is_available()`가 True | 로컬 실행 (2026-11-05) |

## 확인 필요

- PyTorch 공식 설치 페이지(https://pytorch.org/get-started/locally/)에서 안정 버전과 지원 Python 버전 표를 추출하지 못했다. 확인 필요.
- YOLO26 가중치를 받아 실제로 로드되는지는 이 Task에서 확인하지 않았다. 탐지 기준선 단계에서 확인한다.
- Ultralytics 8.4.202와 torch 2.15.0의 호환을 명시한 공식 표는 찾지 못했다. 설치가 해결되고 import가 되는 것까지만 확인했다.
- PyPI 페이지는 크기 때문에 일부만 읽혀 최신 릴리스의 업로드 날짜는 확인하지 못했다.
- Python 3.13 이상으로 올리는 것은 검토하지 않았다. 올릴 때 이 문서를 갱신한다.
