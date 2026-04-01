# Portfolio

--------------------------------------------------------------------------------
# 물리학 기반 현상의 본질을 분석하고 최신 AI 기술로 데이터의 잠재력을 이끌어내는 데이터 사이언티스트, 이종석입니다.

> **"복잡한 물리적/공학적 제약을 정확히 이해하고, LLM과 데이터 분석 능력을 같이 활용하여 도메인 특화 엔지니어링 문제를 주도적으로 해결합니다."**

*   **Point 1:** 비정형 문서(PDF 등) 데이터 추출/정제 및 RAG 파이프라인 고도화 역량 보유
*   **Point 2:** 물리/화학적 공정 원리 이해 및 시뮬레이션-실제 데이터 정합성 검증 경험

---

## 기술 스택

*   **AI & LLM:** LLM API, LangChain, LangFlow, Opik, RAG Pipeline, Prompt Engineering
*   **Data & Pipeline:** n8n, PostgreSQL, OCR, Vector DB 
*   **Engineering & Domain Tools:** Python, SQL, Lumerical (FDTD Simulation), Origin, MATLAB, AutoCAD

---

## Projects

### 1. 감사 정보 통합 관리 및 감사 보고서 초안 작성 서비스 (A.I.D)
**Role:** Leader / LLM Application Developer / Data Engineer | **Period:** 2025.11 - 2026.01
> **한줄 요약:** 비정형 감사 보고서(PDF)를 정제하여 질의응답 및 초안 작성을 돕는 RAG 기반 웹 서비스 구축

**[Problem] 비정형 데이터의 검색 비효율 및 감사 판단의 비일관성**
*   감사 보고서가 PDF/HWP 등 비정형 포맷으로 산재되어 있어 정보 검색에 과도한 시간과 자원이 소모.
*   다양한 감사 사례들과 복잡한 규정을 모두 고려해야 하는 감사 판단의 어려움

**[Action] 데이터 파이프라인 자동화 및 검색 정확도 고도화**
*   **ETL 파이프라인:** 감사 보고서 크롤링 및 적재 → OCR 텍스트 추출 → 메타데이터 정의(감사 사례, 감사 처분) 및 엔티티 추출 AI agent 활용 → DB 적재의 전 과정 자동화 (n8n 활용).
*   **데이터 정제:** 감사 보고서에서 엔티티를 추출하는 과정을 보고서 개요 정보 추출 -> 보고서 목차 분석 및 문서 분할 -> 세부 감사 사례 정보 추출 3 단계의 AI agent 활용
*   **RAG 최적화:** 단순 벡터 검색의 한계를 보완하기 위해 **Hybrid Search (Keyword + Vector)**와 **Multi-Query** 기법 도입.
*   **재정렬(Rerank):** Reranker를 적용하여 검색 결과의 관련성을 재조정, 문맥 파악 능력 강화.
*   **평가 시스템:** `Opik`과 `Ragas`를 도입하여 답변의 Hallucination,Faithfulness,Answer Relevancy,Answer Relevance 를 모니터링.

**[Result] 신뢰 가능한 도메인 특화 챗봇 구현**
*   **정량적 성과:** Ragas 평가 지표 기준 **Answer Relevance 0.97**, **Faithfulness 0.96** 달성.
*   **성능 개선:** Answer Relevancy를 0.62 -> 0.77 로 획기적으로 개선
*   **의의:** 설명 가능한 AI(XAI) 구축 및 구현

<br>

### 2. 소상공인 폐업 예측 및 AI 컨설팅 솔루션 
**Role:** Leader / Data Analyst / ML Engineer | **Period:** 2025.09 - 2025.10
> **한줄 요약:** 상권 빅데이터 분석(XGBoost)과 생성형 AI를 결합하여 폐업 위험 원인을 진단하고 해결책을 제시

**[Problem] 수치 데이터 예측 결과의 낮은 해석 가능성**
*   기존 머신러닝 모델은 폐업 확률(Score)만 제공할 뿐, "왜" 위험한지에 대한 구체적인 설명이 부족하여 실질적 솔루션 제공 불가.

**[Action] XAI(설명 가능한 AI)와 LLM의 결합**
*   **예측 모델링:** 서울 성동구 상권 데이터를 전처리하고 LGBM 앙상블 모델로 폐업 예측 수행.
*   **원인 분석:** **SHAP Value**를 활용하여 예측 결과에 가장 큰 영향을 미친 변수(임대료, 유동인구, 동종 업계 등)를 추출.
*   **자동화 리포팅:** 추출된 위험 요인 데이터를 AI Agent로 맞춤형 경영 개선 컨설팅 보고서 자동 생성.

**[Result] 분석 결과의 설명력(Explainability) 확보**
*   **모델 성능:** F1 score 0.6 달성.
*   **리스크 대응:** 예상 폐업 시일 9개월 전 부터 폐업 경고 및 컨설팅 메시지 제공

<br>

### 3. 초박막 WS2 마하 젠더 변조기의 변조 특성 연구 
**Role:** Researcher (Semiconductor Device Physics) | **Period:** 2022.05 - 2024.02
> **한줄 요약:** 공정 데이터 분석을 통해 결함(Residue) 원인을 규명하고 집적화된 소자 제작 성공

**[Problem] 나노 단위 공정 잔여물로 인한 수율 저하**
*   20nm 두께의 초박막 WS2 마하-젠더 변조기 제작 시, 리소그래피 공정 후 발생하는 미세 잔여물로 인해 소자 성능 저하 발생.

**[Action] 마하 젠더 변조기 동작 및 원리 연구**
*   **공정 최적화:** Soft bake 온도를 낮춰 감광액의 경화를 방지 및 화학적 잔여물 제거.
*   **변조 동작:** 레이저를 사용해 발광에 따른 소자의 온도 변화를 이끌어 내어 변조를 제어 및 변조 성능 측정
*   **시뮬레이션 검증:** `Python`과 `Lumerical(FDTD)`을 활용해 온도에 따른 굴절률 변화와 전파 성능을 시뮬레이션하여, 실제 실험 데이터와 시뮬레이션 결과와의 모델 정합성 확보.

**[Result] 초소형 광 반도체 제작 성공**
*   **성과:** 기존 소자에 비해 구동 길이를 **100 μm에서 2 μm**로 단축하여 신호 전파 및 변조 및 소자 집적화 성공.
*   **논문:** 연구 결과를 정리하여 **Nature Communications**에 제1저자로 게재.

---

## Education

*   **고려대학교 일반대학원 물리학과 석사** (2021.03 ~ 2023.08)
    *   세부전공: 양자광학 / 반도체 나노 소자 공정 및 물성 분석
*   **고려대학교 물리학과 학사** (2015.03 ~ 2021.02)

---

## Links (보고서)
* **감사 정보 통합 관리 및 초안 작성 서비스 (A.I.D)**
   * [**보고서**](https://github.com/Cocou0770/Portfolio/blob/main/AID%20%EC%84%9C%EB%B9%84%EC%8A%A4.pdf)
   * [**시연영상**](https://youtu.be/5OOfkYR8HvY)
* **소상공인 폐업 예측 및 AI 컨설팅 솔루션**
   * [**보고서**](https://github.com/Cocou0770/Portfolio/blob/main/%EA%B0%80%EB%A7%B9%EC%A0%90%20%EC%9C%84%EA%B8%B0%20%EB%B6%84%EC%84%9D%20%EC%8B%9C%EC%8A%A4%ED%85%9C.pdf)
   * [**코드**](https://github.com/Cocou0770/Portfolio/blob/main/%EA%B0%80%EB%A7%B9%EC%A0%90%20%EC%9C%84%EA%B8%B0%20%EB%B6%84%EC%84%9D%20%EC%BD%94%EB%93%9C/notebooks/Baseline.ipynb)

## Paper
* [**Ultrathin WS2 Polariton Waveguide for Efficient Light Guiding**](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adom.202300069)
* [**Ultra-compact exciton polariton modulator based on van der Waals semiconductors**](https://www.nature.com/articles/s41467-024-46701-1)
