# LFWkNN

## 1. 프로젝트 소개

---

### 진행 기간

- 2020년 11월 10일 ~ 2020년 12월 1일

### 참여 인원

- 김인조
- 안승섭

### 프로젝트 진행 배경

- 학과 수업 '심화기계학습'의 팀프로젝트의 일환으로 기존의 알고리즘을 보완하여 더 나은 알고리즘을 설계

## 2. 연구 목적

---

### 2-1) 연구 배경

- `kNN(k Nearest Neighbor`)은 non parametric, supervised learning 알고리즘으로 특정 데이터와 가까운 k개의 데이터를참고하여 해당 데이터의 특성을 할당하는 알고리즘

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/931ea1aa-3253-4111-a54f-0207a6710f58/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/931ea1aa-3253-4111-a54f-0207a6710f58/Untitled.png)

- 위와 같은 경우, k를 1로 하면 파란 원으로 분류 되지만 k를 4로 확장하면 빨간 삼각형으로 분류
- k의 값에 따라 분류성능이 달라지므로 적절한 k값을 찾는 것은  매우 중요

### 2-2) 연구 동기

- kNN 알고리즘의 성능을 높이는 방법은 크게 다음 두가지로 분류
    - Local kNN
    - Global kNN
- Local kNN은 각 데이터마다 다른 k를 할당하는 방법
- Global kNN은 데이터마다 feature weight를 구한후 이를 바탕으로 k를 결정

### 2-3) 연구 목적

- Global kNN 방법 중 하나인 `DCT-kNN` 알고리즘과 `Local kNN`을 결합한 새로운 kNN 알고리즘을 제안

## 3. 필요 기술

---

- 머신러닝 알고리즘 (kNN)에 대한 수학적 이해
- python 코딩 능력
