# YOLO26 원판 fine-tuning·시각화 실험계획

2026-11-18. 실행계획이며, 아래 도구 설치·계정 연결·학습 실행은 아직 하지 않았습니다.

## 목표와 고정 조건

카메라 쪽 원판의 합의한 보이는 외곽을 탐지합니다. 별도 원판이나 겉의 작은 원판만 찾는 오류를 구분합니다.
YOLO26n을 고정하고, 기본 fine-tuning을 먼저 한 뒤 한 번에 한 조건을 비교합니다.
기존 사람 단위 분할을 유지합니다. 학습은 원판 있음 39장 + 없음 6장 = 45장(8명·8편),
검증은 원판 있음 39장(6명·6편)입니다. 데이터 버전과 실제 개수를 실행 전에 재검사합니다.
현재 검증셋은 설정 선택용이며 최종 독립 테스트셋이 아닙니다.
서로 인접한 영상 프레임을 독립 표본으로 간주하지 않습니다.

사전학습 위치 기준선은 클래스 무시·conf 0.25·IoU 0.5에서 TP 11, FP 66, FN 28,
정밀도 14.29%, 재현율 28.21%, N/A 0입니다. 상세 조건은
[coco-location-baseline.md](coco-location-baseline.md)를 참고합니다.
이 결과는 원판 클래스 인식 성능이 아닙니다. fine-tuning 후 plate 출력의 변화와
COCO의 모든 클래스 후보를 센 이전 결과의 의미 차이를 명시합니다.

시각 검수 근거는 [프레임별 관찰 기록](yolo26-visual-review.md)에 보존합니다.

## 학습 전 확정할 비교·선택 규칙

### 기준선 구분

B0는 현재 COCO 사전학습 모델의 클래스 무시 위치 기준선입니다.
E0는 plate 한 클래스로 처음 fine-tuning한 모델입니다.
B0→E0는 학습과 출력 클래스 구성이 모두 달라지는 비교이므로 정밀도 상승을
학습 기법이나 구조 변경만의 효과로 해석하지 않습니다.
이후 증강·학습 조건의 효과는 E0 또는 명시적으로 고정한 직전 기준 run과 비교합니다.

### 체크포인트와 실험 선택

각 run의 체크포인트는 검증 AP50:95가 가장 높은 epoch로 선택합니다.
동률이면 더 이른 epoch를 택합니다. 마지막 epoch나 train loss 최저점으로 임의 선택하지 않습니다.
실제 라이브러리의 best.pt 선택 규칙이 다르면 epoch별 지표와 체크포인트를 보존해
이 규칙으로 선택한 파일을 별도 식별합니다. 평가기·라이브러리 버전도 고정합니다.

조건 간 주지표는 선택된 체크포인트의 검증 AP50:95입니다.
AP50와 conf=0.25·IoU=0.5의 TP/FP/FN·P/R/F1·프레임당 FP는 보조로 함께 보고합니다.
주지표가 동률이거나 차이가 seed 변동에 비해 작으면 개선을 확정하지 않고 단순한 E0를 유지합니다.
실험 도중 유리한 지표로 선택 기준을 바꾸지 않습니다.

### 반복 seed 비교

E0는 fliplr=0, E1은 fliplr=0.5로 두고 다른 설정·데이터·검증 조건을 같게 합니다.
처음에는 같은 seed로 동작을 확인하되, flip 효과를 채택하기 전 E0와 E1 모두
동일한 seed 집합으로 반복합니다. 선택된 E1만 반복하지 않습니다.
seed 집합과 반복 횟수는 반복 실행 전에 팀원과 확정합니다.
각 seed별 E1−E0 차이, 평균·표준편차와 원자료를 함께 보고하며 작은 표본에서 유의성을 단정하지 않습니다.
같은 seed를 써도 augmentation에 따른 난수 경로와 장치 비결정성 때문에 완전 동일성은 보장되지 않습니다.

## 지금 적용할 항목과 문제별 대응

### 지금 적용

- 데이터/라벨/가중치 해시, 소스 commit, 환경·장치와 전체 설정을 run별로 기록합니다.
- 원본과 증강 입력의 bbox 정합, 원판 크기별 분포, split 누수와 빈 라벨 처리를 확인합니다.
- B0와 E0를 분리하고 checkpoint 규칙과 고정 conf 지표/AP 평가를 구분합니다.
- 학습 loss·검증 지표·실제 LR·IoU와 실패 사례를 Matplotlib으로 기록합니다.
- 한 번에 한 조건을 변경합니다. E1을 채택하려면 E0도 동일 seed 집합으로 반복합니다.
- 학습 오류를 분석하기 위해 train 일부를 무증강 eval로 점검할 수 있지만 모델 선택은 validation으로 합니다.

### 확인된 문제에만 적용

| 관찰된 문제 | 먼저 확인 | 그다음 실험 후보 | 확인할 결과 |
| --- | --- | --- | --- |
| loss가 NaN·발산 | 이미지/박스 유효성, 실제 optimizer·LR, 수치 정밀도 | 데이터 오류 수정 후 낮은 LR 또는 precision 설정 한 축 변경 | loss 유한성·재현성 |
| train에서도 거의 못 찾음 | class/path, 변환 bbox, gradient와 업데이트, 소수 train 표본 과적합 점검 | 학습률·동결 범위·학습 길이를 한 축씩 점검 | train 무증강 검출과 val 지표 |
| train은 좋고 val은 나쁨 | 사람·배경 차이와 동일 조건의 train/val 평가 | 학습 인원·배경 다양화, 적절한 증강, 조기 종료·동결/정규화 비교 | 사람별 val 오류와 seed 변동 |
| 원판 근처 예측이 없음 | raw score·출력 수집 threshold, 입력에서 원판 크기 | 학습 인원의 유사 자세 보강 또는 입력 크기 비교 | 고정 conf 재현율·FN |
| 작은 원판·가림에서 실패 | 정답 대상/보이는 외곽 일관성, letterbox 후 bbox 크기 | 학습 데이터 보강 후 해상도 한 축 비교 | 크기·가림별 TP/FN 및 속도 |
| 겉의 작은 원판만 감쌈 | 큰 원판 외곽 라벨과 부분 가림 기준 | 같은 유형의 정답 예시 보강 | IoU·범위 오류 수 |
| 세워둔 원판·배경에 오탐 | 우리의 목표가 '카메라 쪽 장착 원판 하나'인지 확인 | 학습 쪽 배경/비대상 원판 장면과 정확한 target 라벨 추가 | FP 유형·위치와 재현율 |
| 낮은 confidence에만 맞는 박스 | bbox는 맞는지, confidence 분포·PR | validation에서 threshold 선택 | FP/FN trade-off, 기존 0.25 결과도 보존 |
| 메모리 부족·속도 저하 | batch/입력/장치/로그·warmup 측정 | batch 축소, 필요 시 해상도·정밀도 별도 비교 | 메모리와 median/p95 및 검출 성능 |
| 두 조건 차이가 작음 | 데이터·seed·장치와 checkpoint 선택 동일성 | 양쪽 seed 반복, 필요하면 검증 인원 확충 | seed별 차이와 변동성 |
| 원판 없음 오탐 평가 필요 | 기존 빈 라벨은 학습용임 | 미학습 인원의 음성 자료 확보 | 프레임당 FP·FP 발생 프레임 비율 |
| 검수 사례가 많아 탐색이 어려움 | 기존 CSV 오류 목록과 로컬 뷰어의 병목 | FiftyOne 로컬 연결 | 태그·IoU·FN/FP 필터 사용성 |

목표가 '장착 원판 하나'이므로 비대상 원판 예시는 같은 이미지에 목표 원판이 있으면
그 목표를 라벨링해야 합니다. 단순히 모든 박스를 지운 음성 이미지로 만들지 않습니다.
threshold 변경은 학습 모델 개선과 별개로 기록하고 고정 conf 기준선 수치를 대체하지 않습니다.
추가 자료는 학습/검증/독립 테스트 목적과 사람 단위 분할을 먼저 정합니다.
새로운 여러 장면으로 평가 범위를 넓히면 기존 39장 점수와 별도로 보고합니다.
모델 구조나 설명 도구를 바꾸는 일은 첫 오류 대응 수단으로 삼지 않습니다.

## 전처리와 증강

| 요청 항목 | 적용 계획 | 확인 사항 |
| --- | --- | --- |
| Resize | 종횡비를 유지하는 YOLO letterbox, 입력 640부터 시작 | 세로 영상을 정사각형으로 늘여 원판 모양을 왜곡하지 않음. 박스도 같은 좌표 변환 |
| ToTensor | Ultralytics 내부 텐서 변환 사용 | 외부 torchvision 처리를 겹치지 않음. HWC/CHW·dtype·BGR/RGB 변환 확인 |
| Normalize | 설치된 detection trainer의 float /255로 [0,1] 처리 | ImageNet mean/std 정규화나 /255를 중복 적용하지 않음 |
| RandomHorizontalFlip | 기본 학습 이후 비교할 첫 증강 후보: p=0 대 p=0.5 | 이미지와 bbox를 함께 반전. 원본 라벨과 near_side는 수정하지 않음 |
| 밝기·색상 변동 | 배경·조명 오류가 남으면 제한된 강도의 후속 실험 | 임의의 큰 변화가 원판 경계 판단을 망가뜨리지 않는지 시각 확인 |
| scale·translation | 작은 물체·위치 일반화 문제에서 후속 검토 | target 잘림과 bbox clip 정책 확인 |
| 수직 반전·큰 회전 | 첫 실험에서 사용하지 않음 | 역도 측면 동작과 맞지 않는 장면을 만들 수 있음 |
| Mosaic·MixUp·CutMix | 첫 통제 실험에서는 비활성화 제안 | 바벨·원판 관계가 깨지는지 확인 후 각각 별도 검토 |

Resize·ToTensor·Normalize는 전처리이며 랜덤 증강과 구분합니다.
고정 전처리는 train/validation에서 일관되게 적용하되 무작위 증강은 train에만 적용합니다.
학습 실제 설정 전체를 기록하고, 비교 대상 이외의 기본 증강도 명시적으로 고정합니다.
현재 flip 실험은 원판 detection 전용입니다. 추후 pose·좌우 관절·near_side 해석에
재사용할 때는 좌우 대응 규칙을 별도 설계해야 합니다.

## Matplotlib으로 남길 필수 자료

그래프와 그 원천 CSV/JSON을 run별로 함께 보존합니다. 모델 버전, 데이터 해시,
설정, 단위, 표본 수, 매칭 규칙, confidence를 표시합니다.

| 자료 | 용도와 해석 |
| --- | --- |
| train/val loss curve | 해당 버전이 실제 제공하는 box·cls 등 loss별 곡선. 존재하지 않는 DFL loss는 만들지 않음 |
| train vs val | 검증 정체·악화와 학습 손실 하락을 함께 확인. train 증강·모드가 달라 loss 차이만으로 과적합 단정 금지 |
| 정밀도·재현율·AP50·AP50:95의 epoch 곡선 | 일반 분류 accuracy curve 대신 사용. 단일 클래스에서는 AP와 클래스 평균 mAP가 같음 |
| Learning rate curve | optimizer parameter group별 실제 LR과 epoch/step 기록. warmup·scheduler 확인 |
| PR curve 및 AP | confidence를 넓게 훑어 trade-off 확인. 고정 conf 0.25로 잘린 예측만으로 전체 PR/AP를 계산하지 않음 |
| Precision/Recall/F1 vs confidence | 운용 threshold 선택 보조. 선택은 검증셋에서 하고 최종 테스트에는 고정 |
| 탐지용 confusion matrix | plate/background의 매칭·미검출·오탐을 표시. 행/열 방향 명시, background-background TN은 정의하지 않음 |
| 탐지 report 표 | TP·FP·FN, P·R·F1, AP, GT 수, 대상/평가/N/A·사유. sklearn classification_report를 검출 박스에 그대로 적용하지 않음 |
| IoU histogram | 매칭된 박스의 분포와 GT별 최고 IoU 분포를 구분. 후자는 무예측 GT도 0으로 포함해 생존 편향 방지 |
| Confidence histogram | TP와 FP의 신뢰도 분포, 표본 수. FN에는 예측 신뢰도를 임의 부여하지 않음 |
| 오류 분포 막대그래프 | A/B/C, 영상·사람, ready/transition/drive/catch/finish/사이 프레임별 GT·TP·FN 수 |

운용 지표는 기존 기준선처럼 conf 0.25·IoU 0.5로 별도 계산합니다.
AP/PR은 낮은 수집 threshold로 별도 예측을 모으고, 수집 하한·NMS·max_det·매칭 방식도 기록합니다.
Ultralytics 요약 P/R은 최대 F1 지점 등 다른 threshold를 쓸 수 있어 고정 threshold 지표와 열을 분리합니다.
자체 평가기와 COCO/Ultralytics 평가기의 매칭·집계 방식이 다르면 같은 이름으로 섞지 않습니다.

## 추가하면 좋은 지표

- AP75와 IoU:0.50:0.95: 박스가 원판 외곽에 정확히 맞는지 확인합니다.
- 원판 중심 오차: GT 원판 지름 대용값 또는 bbox 대각선으로 정규화하고 정의를 고정합니다.
  미검출을 제외한 오차만 좋게 보이지 않도록 검출률과 함께 표시합니다.
- 대상 범위 오류: 큰 원판 대신 겉의 작은 원판만 잡은 사례와 세워둔 원판을 잡은 사례를 텍스트 태그로 관리합니다.
- 속도·자원: warmup 별도, 추론 median/p95, 전처리/디코딩/후처리 포함 시간 구분,
  모델 파라미터 수·가중치 크기·실행 장치와 메모리 지표의 측정 방법 기록.
- 독립 원판 없음 영상 확보 후 프레임당 FP와 하나 이상 FP가 생긴 프레임 비율.
  현재 빈 라벨 6장은 학습용이므로 독립 성능으로 보고하지 않습니다.
- 조건 채택 전에 기준과 후보 모두 동일한 seed 집합으로 반복해 변동성과 차이를 기록합니다.
  지금 검증 인원 6명만으로 안정적인 일반화나 작은 성능 차이를 단정하지 않습니다.

일반 accuracy, ROC/AUC, specificity는 현재 검출 과제의 주지표에서 제외합니다.
배경 negative의 단위와 TN 개수가 정의되지 않아 그대로 계산하면 해석이 달라집니다.
나중에 독립 영상에서 프레임 단위 '원판 있음/없음'을 별도 정의할 때 ROC 적용을 재검토합니다.

## 현재 실험의 범위

원판 detection의 학습 안정성, 미검출, 오탐, 박스 위치를 판단하는 데 필요한 자료만 구축합니다.
Grad-CAM, feature/activation map, filter visualization은 현재 계획에서 제외합니다.
ROC, 일반 accuracy, 분류 전용 classification report도 만들지 않습니다.
단계별 미검출은 detection 결과를 단계별로 나눈 분석이지 별도 동작 분류기 평가가 아닙니다.

## 실험 도구

| 도구 | 적용 단계와 역할 |
| --- | --- |
| Matplotlib + CSV/JSON | 1차 필수. loss·LR·PR·분포·오류 요약을 재생성 가능한 코드로 출력 |
| Weights & Biases | 1차 학습 기록 도입. 우선 offline, 수치·설정·안전한 순수 그래프만 명시적으로 로깅 |
| FiftyOne | 첫 평가 후 반복적인 사례 탐색이 필요할 때만 로컬에서 도입. FP/FN·IoU·단계별 필터링, 기존 data/derived 프레임 참조 |
| 데이터·설정 해시와 Git commit | run 재현에 필수. 초기에는 별도 대형 데이터 관리 플랫폼 없이 manifest와 SHA256로 시작 |

W&B의 자동 Ultralytics 통합은 예측 이미지·모델 artifact를 기록할 수 있어 그대로 켜지 않습니다.
수치 전용 adapter와 로그 검사를 먼저 구현합니다. offline은 나중에 자동 sync하지 않습니다.
온라인 사용은 업로드 항목을 확인하고 별도로 연결하며 이미지·가중치 업로드는 하지 않습니다.
FiftyOne은 localhost에서만 띄우고 cloud/publish/export는 사용하지 않습니다.
서로 다른 도구의 FP/FN은 threshold와 매칭 정의를 맞춰 교차검증합니다.
Netron은 현재 구조 변경을 하지 않으므로 이번 도구 도입 범위에서 제외합니다.

## 실험 순서와 구현 단위

1. 데이터셋·로그 계약: Task 6b와 연결해 동일 영상·프레임·split·라벨·출력 경로와 해시 고정.
   기존 data/README 정책에 따라 학습 이미지 생성은 data/derived 아래에서만 수행.
2. 짧은 학습 점검: 변환된 이미지+bbox를 메모리로 확인하고, optimizer step·loss 유한성·체크포인트 저장 점검.
3. E0: 기본 YOLO26n plate fine-tuning. fliplr=0, 복잡한 증강 없이 시작하는 통제 학습 제안.
   optimizer·LR·batch·epochs·patience·freeze·seed 전체를 구현 전 팀원과 고정.
   optimizer=auto가 수동 LR을 덮는지 확인하고 수동 LR 비교에서는 optimizer도 명시.
4. E1: E0와 같은 조건에서 horizontal flip만 p=0.5로 변경. 데이터셋/검증셋은 그대로 유지.
5. 오류 분석 뒤 E2: 동결 범위 또는 LR 등 한 축만 변경. 헤드 구조 변경은 근거가 생긴 후.
6. 조건 채택 전 E0와 E1 양쪽을 같은 seed 집합으로 반복한 뒤 판단. 독립 테스트셋과 원판 없음 검증 자료 확보.
7. Matplotlib의 프레임별 오류 목록과 기존 로컬 뷰어로 검수합니다.
   반복 탐색이 불편해지면 FiftyOne을 연결합니다. 내부 특징 시각화는 구현하지 않습니다.

학습 실행·도구 설치·계정 연결·새로운 이미지 저장 정책 변경은 이 계획 작성에 포함하지 않습니다.
Matplotlib 기록과 재현 가능한 기본 학습·평가를 먼저 완성합니다.

## 검증과 완료 기준

- 합성 예측·정답으로 TP/FP/FN·IoU·PR 계산을 확인하고 고정 conf 지표와 AP 수집을 분리합니다.
- 좌우 반전 bbox 변환과 두 번 반전 시 복원, 빈 라벨 유지, 검증 무증강을 테스트합니다.
- 표의 수치와 Matplotlib 곡선이 동일 원천 기록에서 나오고 원천 파일을 함께 보존합니다.
- run마다 train/validation 인원·영상·GT 수, seed 적용 여부, 설정·데이터·가중치 해시를 기록합니다.
- W&B/FiftyOne smoke test에서 원본 이미지·가중치가 외부로 전송되지 않는지 확인합니다.
- Git에는 코드·표·안전한 문서만 올립니다. 순수 지표 그래프도 공유 전 이미지가 포함되지 않았는지 확인합니다.
- 자동 plot 저장이 train batch나 val 예시 얼굴 이미지를 만들지 않게 설정/콜백을 확인합니다.
- 학습 그래프는 Git 제외 artifacts의 run별 폴더에, 영상 기반 표시는 저장하지 않고 메모리에서 확인합니다.
- 새 코드는 전체 테스트 및 새 기능 합성 테스트를 통과해야 합니다.

## 근거 문서

- Ultralytics 전처리·bbox 동기 변환: https://docs.ultralytics.com/reference/data/augment/
- 탐지 지표·곡선: https://docs.ultralytics.com/guides/yolo-performance-metrics/
- 평가 confidence와 AP: https://docs.ultralytics.com/usage/cfg/
- Fine-tuning: https://docs.ultralytics.com/guides/finetuning-guide/
- W&B 통합: https://docs.ultralytics.com/integrations/weights-biases/
- FiftyOne 로컬 탐색: https://docs.voxel51.com/user_guide/app.html
