# Dense Passage Retrieval

## Dense Passage retrieval이란?
DPR은 자연어 처리 분야에서 사용되는 검색 기술 중 하나로 써 문서 검색 시스템에서 효과적으로 사용됩니다.
<br>
<br>
DPR은 문서를 벡터화한 후, 밀도가 높은 (dense) 벡터 공간에서 유사도를 계산을 수행합니다. 이를 통해 사용자가 입력한 검색어와 가장 유사한 문서를 찾아냅니다. 이러한 밀도가 높은 벡터 공간을 만드는 데에는 인공 신경망 기술 중 하나인 양방향 인코더가 사용됩니다.

## DPR이 기존 검색기술인 TF-IDF기반의 검색과 다른 점 
TF-IDF는 단어의 빈도수와 연무서 빈도를 이용하여 검색어와 문서 간의 유사도를 계산하는 기술입니다. 이 방법은 검색어와 문서 간의 단어를 매칭을 고려하며, 단어의 빈도수에 가중치를 두어 중요한 단어를 강조합니다.
<br>
<br>
BM25은 TF-IDF 기술을 보안하여 고안된 검색 기술입니다. BM25는 단어 빈도수와 문서길이, 검색어의 단어 간 거리등을 고려하여 유사도를 계산합니다. 이를 통해 검색 결과의 질을 더욱 향상시킬 수 있습니다.
<br>
<br>
따라서, 이들 검색 기술의 가장 큰 차이점은 유사도를 계산하는 방법입니다. TF-IDF와 BM25 는 단어의 빈도수를 고려한 기술이고, DPR은 벡터화된 문서간의 유사도를 계산합니다. 그리고 DPR은 대규모 데이터셋에서 빠른 검색 속도와 높은 검색 성능을 보장하는 데 특화되어있습니다.

## 실행환경
Pycharm community 툴을 사용하였습니다.

## 실행방법
1. 시스템 환경변수에서 venv 폴더를 환경변수로 설정
2. setup으로 필요한 패키지 설치
3. set flask_app=server
4. 기본 포트 5000 방화벽에서 인바운드 규칙 설정 포트 5000번연결 허용
5. flask run --host=0.0.0.0 --with-threads
<br>
주의: 아이템이 많은 데이터는 시간이 오래걸리므로 sentnece20과 같은 작은 사이즈의 json파일을 업로드 해주세요.

## 개선점:

- 모델 압축 데이터 추가 <br>
- 고유 url을 주기 위해 기존에 ip에 접속 시간까지 추가
