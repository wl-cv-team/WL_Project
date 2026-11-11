# wl_snatch

인상(snatch) 영상에서 측정 가능한 것을 신뢰할 수 있게 측정하고, 그 데이터를 쌓아 나중에 분석하는 연구 저장소입니다.

- 지향점과 표현 규칙: [docs/vision.md](docs/vision.md)
- 단계 순서: [docs/roadmap.md](docs/roadmap.md)
- 환경 결정 기록: [docs/decisions/0001-environment.md](docs/decisions/0001-environment.md)
- 문헌 목록: [docs/literature/README.md](docs/literature/README.md)
- 설계 문서: [docs/superpowers/specs/](docs/superpowers/specs/)

## 설치

[uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync
uv run pytest
```

## 구조

- `src/wl_snatch/`: 모든 로직(.py 모듈)
- `tests/`: 테스트. 노트북은 실행하지 않습니다.
- `scripts/`: 모듈을 호출만 하는 실행기
- `tutorials/`: 모듈을 불러다 쓰는 튜토리얼과 실험
- `data/`: git 제외. [data/README.md](data/README.md) 참고

## 협업

`main`에 직접 push하지 않습니다. `feature/<주제>` 브랜치에서 작업하고 PR로 상호 리뷰합니다. 데이터의 외부 공유는 사전 승인을 받습니다.
